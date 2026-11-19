"""Run the dispatcher: python3 -m dispatcher --db switchyard.db --port 7000"""

import argparse

from . import __version__
from .dispatch import Dispatcher
from .server import make_server


def main(argv=None):
    parser = argparse.ArgumentParser(prog="dispatcher", description="Switchyard webhook dispatcher")
    parser.add_argument("--db", default="switchyard.db", help="SQLite file (created if missing)")
    parser.add_argument("--port", type=int, default=7000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--log", default="dispatcher.log", help="log file (appended to)")
    args = parser.parse_args(argv)

    dispatcher = Dispatcher(args.db, log_path=args.log)
    server = make_server(dispatcher, args.host, args.port)
    dispatcher.log.write("INFO", "started version=%s port=%d db=%s" % (__version__, server.server_port, args.db))
    print("dispatcher %s listening on %s:%d" % (__version__, args.host, server.server_port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        dispatcher.log.write("INFO", "stopped")
        dispatcher.close()


if __name__ == "__main__":
    main()
