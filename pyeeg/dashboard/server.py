"""
Dashboard server entry point.

This module provides the main entry point for starting the pyEEG dashboard server.
"""

import argparse
import logging
from .app import create_app
from .._logging import LOGGER

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    """Main entry point for the dashboard server."""
    parser = argparse.ArgumentParser(
        description='pyEEG Dashboard Server - Interactive TRF Analysis'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host address to bind to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port to listen on (default: 5000)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='Enable debug mode'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Create app
    app = create_app()
    
    LOGGER.info(f"Starting pyEEG Dashboard Server")
    LOGGER.info(f"Host: {args.host}")
    LOGGER.info(f"Port: {args.port}")
    LOGGER.info(f"Debug mode: {args.debug}")
    LOGGER.info(f"Log level: {args.log_level}")
    
    # Run server
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )


if __name__ == '__main__':
    main()
