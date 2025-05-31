"""User interaction utilities."""

import logging
from src.server.schemas import UserConfirmation

logger = logging.getLogger(__name__)

class UserPrompts:
    """Centralized user prompt handling."""
    
    @staticmethod
    def get_confirmation(prompt: str, action: str) -> UserConfirmation:
        """
        Gets user confirmation with type safety. Keeps asking until valid input.
        Supports 'yes', 'no', 'exit', and 'quit' responses.
        """
        while True:
            try:
                user_input = input(f"{prompt} (yes/no/exit): ").strip().lower()
                if user_input == 'yes':
                    return UserConfirmation(
                        confirmed=True,
                        action=action,
                        message=f"User confirmed: {action}"
                    )
                elif user_input == 'no':
                    return UserConfirmation(
                        confirmed=False,
                        action=action,
                        message=f"User declined: {action}"
                    )
                elif user_input in ['exit', 'quit']:
                    return UserConfirmation(
                        confirmed=False,
                        action="exit",
                        message="User chose to exit"
                    )
                else:
                    print("Invalid input. Please type 'yes', 'no', or 'exit'.")
            except (KeyboardInterrupt, EOFError):
                print("\nOperation cancelled by user.")
                return UserConfirmation(
                    confirmed=False,
                    action="cancelled",
                    message="User cancelled the operation"
                )
    
    @staticmethod
    def show_app_header():
        """Display application header."""
        print("Interactive Chat with RAG API (type 'exit' or 'quit' to end)")
        print("Ensure the FastAPI server is running on http://localhost:8000")
        print("-------------------------------------------------------------")
    
    @staticmethod
    def show_section_separator():
        """Display section separator."""
        print("-------------------------------------------------------------")
    
    @staticmethod
    def show_chat_start():
        """Display chat session start message."""
        print("Starting chat session...")

# Convenience instance
user_prompts = UserPrompts() 