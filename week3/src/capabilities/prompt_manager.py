import os
import re
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PromptArgument:
    name: str
    description: str
    required: bool = True
    default: Any = None
    type: str = "string"


@dataclass
class Prompt:
    name: str
    description: str
    template: str
    arguments: list[PromptArgument] = field(default_factory=list)
    version: str = "1.0.0"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def render(self, **kwargs) -> str:
        for arg in self.arguments:
            if arg.required and arg.name not in kwargs:
                raise ValueError(f"Required argument '{arg.name}' not provided")
        
        result = self.template
        for arg_name, value in kwargs.items():
            placeholder = f"{{{arg_name}}}"
            result = result.replace(placeholder, str(value))
        
        remaining_placeholders = re.findall(r'\{([^{}]+)\}', result)
        for placeholder in remaining_placeholders:
            for arg in self.arguments:
                if arg.name == placeholder and arg.default is not None:
                    result = result.replace(f"{{{placeholder}}}", str(arg.default))
        
        return result


class PromptManager:
    def __init__(self):
        self._prompts: dict[str, Prompt] = {}
        self._prompt_dir: Optional[Path] = None
    
    def register_prompt(
        self,
        name: str,
        description: str,
        template: str,
        arguments: Optional[list[PromptArgument]] = None,
        version: str = "1.0.0"
    ) -> Prompt:
        prompt = Prompt(
            name=name,
            description=description,
            template=template,
            arguments=arguments or [],
            version=version,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        self._prompts[name] = prompt
        return prompt
    
    def register_prompt_from_file(
        self,
        file_path: Path,
        name: Optional[str] = None
    ) -> Optional[Prompt]:
        if not file_path.exists() or not file_path.is_file():
            return None
        
        try:
            content = file_path.read_text(encoding="utf-8")
            
            lines = content.split('\n')
            description = ""
            arguments: list[PromptArgument] = []
            template_start = 0
            
            for i, line in enumerate(lines):
                if line.startswith("---"):
                    continue
                if line.startswith("# Description:"):
                    description = line[len("# Description:"):].strip()
                elif line.startswith("# Arg:"):
                    arg_line = line[len("# Arg:"):].strip()
                    parts = arg_line.split("|")
                    if len(parts) >= 2:
                        arg_name = parts[0].strip()
                        arg_desc = parts[1].strip()
                        arg_required = True
                        arg_default = None
                        
                        if len(parts) >= 3:
                            required_part = parts[2].strip()
                            if required_part.lower() == "optional":
                                arg_required = False
                        
                        if len(parts) >= 4:
                            arg_default = parts[3].strip()
                        
                        arguments.append(PromptArgument(
                            name=arg_name,
                            description=arg_desc,
                            required=arg_required,
                            default=arg_default
                        ))
                elif not line.startswith("#") and line.strip():
                    template_start = i
                    break
            
            template = '\n'.join(lines[template_start:])
            prompt_name = name or file_path.stem
            
            return self.register_prompt(
                name=prompt_name,
                description=description or f"Prompt from {file_path.name}",
                template=template,
                arguments=arguments
            )
            
        except Exception:
            return None
    
    def auto_register_prompts(self, prompts_dir: Optional[Path] = None) -> int:
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent.parent.parent / "prompts"
            if not prompts_dir.exists():
                prompts_dir = Path(os.getcwd()) / "prompts"
        
        self._prompt_dir = prompts_dir
        
        if not prompts_dir.exists():
            return 0
        
        registered_count = 0
        
        for file_path in prompts_dir.glob("*.md"):
            if file_path.name.startswith("_"):
                continue
            
            prompt = self.register_prompt_from_file(file_path)
            if prompt:
                registered_count += 1
        
        return registered_count
    
    def unregister_prompt(self, name: str) -> None:
        if name in self._prompts:
            del self._prompts[name]
    
    def get_prompt(self, name: str) -> Optional[Prompt]:
        return self._prompts.get(name)
    
    def render_prompt(self, name: str, **kwargs) -> Optional[str]:
        prompt = self._prompts.get(name)
        if prompt is None:
            return None
        return prompt.render(**kwargs)
    
    def list_prompts(self) -> list[dict[str, Any]]:
        return [
            {
                "name": p.name,
                "description": p.description,
                "version": p.version,
                "arguments": [
                    {
                        "name": arg.name,
                        "description": arg.description,
                        "required": arg.required,
                        "default": arg.default
                    }
                    for arg in p.arguments
                ],
                "created_at": p.created_at.isoformat() if p.created_at else None,
                "updated_at": p.updated_at.isoformat() if p.updated_at else None
            }
            for p in self._prompts.values()
        ]
    
    def update_prompt(
        self,
        name: str,
        template: Optional[str] = None,
        description: Optional[str] = None,
        arguments: Optional[list[PromptArgument]] = None
    ) -> bool:
        if name not in self._prompts:
            return False
        
        prompt = self._prompts[name]
        
        if template is not None:
            prompt.template = template
        if description is not None:
            prompt.description = description
        if arguments is not None:
            prompt.arguments = arguments
        
        prompt.updated_at = datetime.now()
        return True
    
    def clear(self) -> None:
        self._prompts.clear()
