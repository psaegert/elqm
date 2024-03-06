import argparse
import os
import sys

from dynaconf import Dynaconf


def main(argv: str = None) -> None:
    """
    Parse the command line arguments for commands and options
    """

    # Parse the command line arguments for commands and options
    parser = argparse.ArgumentParser(description='Energy Law Query Master')
    subparsers = parser.add_subparsers(dest='command_name', required=True)

    # Scraper and arguments
    scrape_parser = subparsers.add_parser("scrape-data")
    scrape_parser.add_argument('-i', '--interval', type=float, default=0.5, help='Interval between requests')

    # CLI chat
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument('-c', '--config', required=True, type=str, help='Path to the config file')
    run_parser.add_argument('-v', '--verbose', action='store_true', help='Whether to print debug information')

    # GUI chat
    gui_parser = subparsers.add_parser("gui")
    gui_parser.add_argument('-c', '--config', required=True, type=str, help='Path to the config file')
    gui_parser.add_argument('-s', '--verbose', action='store_true', help='Whether to print debug information')
    gui_parser.add_argument('--share', action='store_true', help='Whether to share the frontend')

    # Clear cache
    clear_cache_parser = subparsers.add_parser("clear-cache")
    clear_cache_parser.add_argument('-c', '--config', type=str, help='Path to the config file')
    clear_cache_parser.add_argument('-i', '--index', type=str, help='Name of the index to clear')
    clear_cache_parser.add_argument('-f', '--force', action='store_true', help='Whether to force clear the cache')

    # Evaluate input
    args = parser.parse_args(argv)

    # Execute the command
    match args.command_name:
        case 'scrape-data':
            from elqm.scraper import scrape_data
            scrape_data(interval=args.interval)
        case 'run':
            from elqm.elqm_pipeline.elqm_pipeline import ELQMPipeline

            config = Dynaconf(settings_files=os.path.abspath(args.config))

            elqm_pipeline = ELQMPipeline(config)

            if args.verbose:
                print("Running ELQM pipeline with the following configuration:")
                print(elqm_pipeline)
            print("Ask a question or type /bye to exit or /? or /help for help.")

            # Start a user-interaction loop
            while True:
                question = input(">>> ")

                match question:
                    case '/bye':
                        break
                    case '/reset':
                        elqm_pipeline.clear_chat_history()
                        print("Chat history has been reset.")
                    case '/?' | '/help':
                        print("/bye \t\tExit the application.")
                        print("/reset \t\tReset the chat history.")
                        print("/? or /help \tPrint this help message.")

                    case _:  # Answer the question
                        print(elqm_pipeline.answer(question))

        case 'gui':
            from elqm.elqm_pipeline.elqm_pipeline import ELQMPipeline
            from elqm.frontend.gradio import launch_gradio_frontend

            config = Dynaconf(settings_files=os.path.abspath(args.config))

            elqm_pipeline = ELQMPipeline(config)

            if args.verbose:
                print("Running ELQM pipeline with the following configuration:")
                print(elqm_pipeline)

            launch_gradio_frontend(elqm_pipeline, share=args.share)

        case 'clear-cache':
            from elqm.utils import clear_cache

            if args.config is None and args.index is None:
                print("Please provide a config file or an index name to clear the cache.")
                print("Usage: elqm clear-cache -c <config_file> -i <index_name> [-f]")
                sys.exit(1)

            if args.config is not None:
                config = Dynaconf(settings_files=os.path.abspath(args.config))
                index_name = config.index_name
            else:
                index_name = args.index

            if args.force:
                clear_cache(index_name)
                print(f"Cache for index {index_name} cleared.")
            else:
                confirm = input("Are you sure you want to clear the cache? (y/n) ")
                if confirm.lower() == "y":
                    clear_cache(index_name)
                    print(f"Cache for index {index_name} cleared.")
                else:
                    print("Cache not cleared.")

        case _:
            print('Unknown command: ', args.command)
            sys.exit(1)
