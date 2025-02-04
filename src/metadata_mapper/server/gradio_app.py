""" 

This script is used to create a gradio app for the SciNoBo FoS taxonomy mapping functionality. It takes 
the input and calls the 'search/fos_taxonomy' endpoint from the SciNoBo Metadata Mapper API.

"""

import gradio as gr
import os
import requests

# receive the API IP and port and endpoint from env variables
API_IP = os.getenv("API_IP")
API_PORT = os.getenv("API_PORT")
API_ENDPOINT = os.getenv("API_ENDPOINT")
INDEX_NAME = os.getenv("INDEX_NAME")
INDEX_VERSION = os.getenv("VERSION")

headers = {
    "Accept": "application/json",
    "Authorization": "XXXX",
    "APIKey": "YYYY",
    "Content-Type": "application/json"
}


# Define the functions to handle the inputs and outputs
def analyze_input(
    my_id: str | None,
    snippet: str | None,
    k: str | None,
    approach: str | None,
    progress=gr.Progress(track_tqdm=True)
):
    # convert the k to int
    k = int(k)
    # call the API
    response = requests.post(
        f"http://{API_IP}:{API_PORT}/{API_ENDPOINT}",
        json={"id": my_id, "text": snippet, "k": k, "approach": approach}, 
        headers=headers,
        params={"index": INDEX_NAME, "version": INDEX_VERSION},
        verify=False
    )
    return response.json()


# Define the interface for the first tab (Text Analysis)
with gr.Blocks() as text_analysis:
    gr.Markdown("### SciNoBo Field of Science (FoS) Taxonomy Mapper")
    id_input = gr.Textbox(label="ID", placeholder="Enter an ID for the request. At this demo, it is only used for reference.")
    snippet_input = gr.Textbox(label="Input Text", placeholder="Enter a Field of Science or a keyword or a phrase.")
    k_input = gr.Textbox(label="Num. of results", placeholder="Enter the number of the retrieved results")
    approach_input = gr.Textbox(label="Retrieval Approach", placeholder="Enter the approach for the retrieval (cosine, elastic, hybrid)")
    process_text_button = gr.Button("Process")
    text_output = gr.JSON(label="Output")
    process_text_button.click(analyze_input, inputs=[id_input, snippet_input, k_input, approach_input], outputs=[text_output])
    examples = gr.Examples(
        [
            [
                "1",
                "cancer and tumor",
                "10",
                "cosine"
            ],
            [
                "1",
                "immune cell regulation and tolerance",
                "10",
                "cosine"
            ],
            [
                "1",
                "cellular function and structure",
                "10",
                "cosine"
            ],
            
        ],
        inputs=[id_input, snippet_input, k_input, approach_input]
    )


# Combine the tabs into one interface
with gr.Blocks() as demo:
    gr.TabbedInterface([text_analysis], ["Enter Metadata Mode"])

# Launch the interface
demo.queue().launch(server_name="0.0.0.0", server_port=7860)