"""Command-line administration for the consultancy SaaS."""

import argparse
import json

from .db import Database
from .service import ConsultancyService


def build_parser():
    parser = argparse.ArgumentParser(prog="drmoagi-saas")
    sub = parser.add_subparsers(dest="command", required=True)

    init_db = sub.add_parser("init-db", help="create database tables")
    init_db.add_argument("--database-url")

    serve = sub.add_parser("serve", help="run the SaaS API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)

    bootstrap = sub.add_parser(
        "bootstrap", help="create the first company and administrator"
    )
    bootstrap.add_argument("--company", required=True)
    bootstrap.add_argument("--slug", required=True)
    bootstrap.add_argument("--legal-name", required=True)
    bootstrap.add_argument("--admin-name", required=True)
    bootstrap.add_argument("--admin-email", required=True)
    bootstrap.add_argument("--password", required=True)
    bootstrap.add_argument("--currency", default="ZAR")
    bootstrap.add_argument("--tax-rate-bps", type=int, default=0)
    bootstrap.add_argument("--database-url")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "init-db":
        database = Database(args.database_url)
        database.create_schema()
        print(json.dumps({"status": "ok", "database": str(database.engine.url)}))
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("jarvisx.saas.app:app", host=args.host, port=args.port)
    elif args.command == "bootstrap":
        service = ConsultancyService(Database(args.database_url))
        result = service.bootstrap(
            args.company,
            args.slug,
            args.legal_name,
            args.admin_name,
            args.admin_email,
            args.password,
            args.currency,
            args.tax_rate_bps,
        )
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
