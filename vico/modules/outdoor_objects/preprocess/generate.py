try:
    from gradio_client import Client as GradioClient
except:
    pass 

import requests
import json
import os
import time
import shutil
from typing import Optional

def remove_whitespace(s: str) -> str:
    return s.replace(" ", "_")

class StableDiffusion:
    '''
    Wrapper for the Stable Diffusion model.
    '''

    def __init__(
            self, model_name: str = 'stabilityai/stable-diffusion-2', 
            workspace: str = '/tmp/vico',
            access_token: Optional[str] = None
        ):
        '''
        Initialize the Stable Diffusion model.
        Args:
            model_name (str): The name of the model.
            workspace (str): The directory to store the generated images.
            access_token (str, optional): The access token for the model.
        '''
        self.model_name = model_name
        self.workspace = workspace
        self.access_token = access_token
        if self.access_token is None:
            self.access_token = os.environ.get('HF_ACCESS_TOKEN', None)
        assert self.access_token is not None, "Access token is required."
        
    def __call__(self, input_text: str = '', max_tries: int = 5) -> str:
        '''
        Generate an image for the given text.
        Args:
            input_text (str): The input text.
            max_tries (int): The maximum number of tries.
        Returns:
            str: The path to the generated image.
        '''
        headers = {"Authorization": f"Bearer {self.access_token}"}
        API_URL = f"https://api-inference.huggingface.co/models/{self.model_name}"
        
        n_tries = 0
        while n_tries < max_tries:
            response = requests.post(API_URL, headers=headers, json={"inputs": input_text})
            if response.status_code == 200:
                break
            n_tries += 1
            time.sleep(2 ** n_tries)
        else:
            raise Exception(f"Failed to generate image for text: {input_text}")

        image_data = response.content
        output_image_path = os.path.join(self.workspace, f'{remove_whitespace(input_text)}.png')
        with open(output_image_path, "wb") as f:
            f.write(image_data)
        return output_image_path

class One2345:
    '''
    Wrapper for the One-2-3-45 mesh generation. 
    See https://huggingface.co/spaces/One-2-3-45/One-2-3-45 for more details.
    
    If web api service is not available, use a local model as fallback (TODO).
    '''
    def __init__(self, mode: str = 'hf', workspace: str = '/tmp/vico'):
        '''
        Initialize the One-2-3-45 mesh generation.
        Args:
            mode (str): The mode to use for mesh generation.
        '''
        self.mode = mode
        self.workspace = workspace
        if self.mode in ['hf']:
            self.client = GradioClient("https://one-2-3-45-one-2-3-45.hf.space/")
        else:
            raise ValueError(f"Invalid mode: {mode}") 
        
    def __call__(self, input_img_path: str = '') -> str:
        '''
        Generate a mesh for the given image.
        Args:
            input_img_path (str): The path to the input image.
        Returns:
            str: The path to the generated mesh.
        '''
        if self.mode in ['hf']:
            output_mesh_path = self.client.predict(input_img_path, True, api_name='/generate_mesh')
        return output_mesh_path

def generate_mesh(input_text: str = '', workspace: str = '/tmp/vico') -> str:
    '''
    Generate a mesh for the given text.
    Args:
        input_text (str): The input text.
        workspace (str): The directory to store the generated images.
    Returns:
        str: The path to the generated mesh.
    '''
    stable_diffusion = StableDiffusion(workspace=workspace)
    image_path = stable_diffusion(input_text, max_tries=5)
    
    one2345 = One2345(workspace=workspace)
    mesh_path = one2345(image_path)
    return mesh_path
    
def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Generate mesh from text')
    parser.add_argument('--input_text', type=str, default='a white tent', help='The input text')
    parser.add_argument('--workspace', type=str, default='/tmp/vico', help='The workspace directory')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    mesh_path = generate_mesh(args.input_text, args.workspace)
    print(f"Generated mesh: {mesh_path}")