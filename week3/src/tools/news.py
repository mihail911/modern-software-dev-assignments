import os
import httpx
from typing import Any, Optional
from .base import BaseTool


class NewsTool(BaseTool):
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            name="search_news",
            description="搜索最新新闻文章，支持按关键词、类别、语言等条件筛选"
        )
        self._api_key = api_key or os.getenv("NEWSAPI_KEY")
        self._base_url = "https://newsapi.org/v2"
        self._rate_limit_remaining: Optional[int] = None
        self._rate_limit_reset: Optional[int] = None
    
    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词或短语，用于在新闻标题和内容中搜索"
                },
                "category": {
                    "type": "string",
                    "enum": ["business", "entertainment", "general", "health", "science", "sports", "technology"],
                    "description": "新闻类别，如 business、technology、sports 等"
                },
                "language": {
                    "type": "string",
                    "enum": ["ar", "de", "en", "es", "fr", "he", "it", "nl", "no", "pt", "ru", "sv", "ud", "zh"],
                    "description": "新闻语言代码，如 en（英文）、zh（中文）、es（西班牙文）等",
                    "default": "en"
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["relevancy", "popularity", "publishedAt"],
                    "description": "排序方式，relevancy按相关性，popularity按流行度，publishedAt按发布时间",
                    "default": "publishedAt"
                },
                "page_size": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "返回的新闻数量，最大100条，默认20条",
                    "default": 20
                },
                "page": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "页码，用于分页获取结果",
                    "default": 1
                }
            },
            "required": []
        }
    
    async def execute(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        language: str = "en",
        sort_by: str = "publishedAt",
        page_size: int = 20,
        page: int = 1
    ) -> dict[str, Any]:
        if not self._api_key:
            return {
                "success": False,
                "error": "News API key not configured. Please set NEWSAPI_KEY environment variable."
            }
        
        if not query and not category:
            return {
                "success": False,
                "error": "Please provide either a 'query' or 'category' parameter."
            }
        
        if self._rate_limit_remaining is not None and self._rate_limit_remaining <= 0:
            return {
                "success": False,
                "error": "Rate limit exceeded. Please try again later."
            }
        
        page_size = min(max(1, page_size), 100)
        page = max(1, page)
        
        try:
            params: dict[str, Any] = {
                "apiKey": self._api_key,
                "language": language,
                "sortBy": sort_by,
                "pageSize": page_size,
                "page": page
            }
            
            endpoint = f"{self._base_url}/top-headlines"
            
            if query:
                params["q"] = query
            if category:
                params["category"] = category
            
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(endpoint, params=params)
                
                if "X-RateLimit-Remaining" in response.headers:
                    self._rate_limit_remaining = int(response.headers["X-RateLimit-Remaining"])
                
                if response.status_code == 401:
                    return {
                        "success": False,
                        "error": "Invalid API key. Please check your NEWSAPI_KEY."
                    }
                elif response.status_code == 429:
                    return {
                        "success": False,
                        "error": "Too many requests. Rate limit exceeded, please try again later."
                    }
                elif response.status_code != 200:
                    error_data = response.json() if response.text else {}
                    error_message = error_data.get("message", f"API request failed with status code: {response.status_code}")
                    return {
                        "success": False,
                        "error": error_message
                    }
                
                data = response.json()
                
                if data.get("status") != "ok":
                    return {
                        "success": False,
                        "error": data.get("message", "API returned an error status")
                    }
                
                articles = data.get("articles", [])
                total_results = data.get("totalResults", 0)
                
                formatted_articles = []
                for article in articles:
                    source = article.get("source", {})
                    formatted_article = {
                        "title": article.get("title"),
                        "description": article.get("description"),
                        "content": article.get("content"),
                        "url": article.get("url"),
                        "url_to_image": article.get("urlToImage"),
                        "source": {
                            "id": source.get("id"),
                            "name": source.get("name")
                        },
                        "author": article.get("author"),
                        "published_at": article.get("publishedAt")
                    }
                    formatted_articles.append(formatted_article)
                
                result = {
                    "success": True,
                    "total_results": total_results,
                    "page": page,
                    "page_size": page_size,
                    "articles_count": len(formatted_articles),
                    "articles": formatted_articles
                }
                
                return result
                
        except httpx.TimeoutException:
            return {
                "success": False,
                "error": "Request timed out. Please try again later."
            }
        except httpx.ConnectError:
            return {
                "success": False,
                "error": "Failed to connect to the news API. Please check your internet connection."
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"An unexpected error occurred: {str(e)}"
            }
