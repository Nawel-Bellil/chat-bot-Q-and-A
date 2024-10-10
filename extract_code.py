import json

def extract_code_from_notebook(input_notebook, output_file):
    # Open the Jupyter notebook (.ipynb) file
    with open(input_notebook, 'r', encoding='utf-8') as nb_file:
        notebook_content = json.load(nb_file)

    # Create or open the output file where the code will be saved
    with open(output_file, 'w', encoding='utf-8') as code_file:
        # Loop through each cell in the notebook
        for cell in notebook_content['cells']:
            # Check if the cell is a code cell
            if cell['cell_type'] == 'code':
                # Write each line of the code from the cell into the output file
                code_file.write(''.join(cell['source']))
                code_file.write('\n\n')  # Add spacing between different code cells

    print(f"Code extracted and saved to {output_file}")

# Example usage
input_notebook = 'C:/Users/Morsi Store DZ/ai-vol-2/chatbot.ipynb'  # Replace with your notebook file
output_file = 'code.py'       # The file where you want to save the code
extract_code_from_notebook(input_notebook, output_file)
