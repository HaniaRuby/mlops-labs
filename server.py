import litserve as ls
from mlopslabs.deployment.online.api import InferenceAPI

if __name__ == "__main__":
    # Instantiate her schema lifecycle pattern
    api = InferenceAPI()

    # Build server configuration matrix
    server = ls.LitServer(api, accelerator="cpu")

    # Serve the app endpoint up on internal cluster port 8000
    server.run(port=8000, generate_client_file=False)