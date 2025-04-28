import os
import nbformat

# Define the path to the folder containing your notebooks
folder_path = r'E:\Drive\NOA\MAMOTH-Prediction\predict'

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.ipynb'):  # Only process Jupyter notebooks
        filepath = os.path.join(folder_path, filename)

        # Read the notebook
        with open(filepath, 'r', encoding='utf-8') as file:
            notebook = nbformat.read(file, as_version=4)

        # Loop through all cells in the notebook
        for cell in notebook.cells:
            if cell.cell_type == 'code':  # Check if it's a code cell
                # Replace 'import fs.' with 'import .' in the code cell's source
                cell.source = cell.source.replace('ipynb.fs.defs.', '')

        # Save the modified notebook
        with open(filepath, 'w', encoding='utf-8') as file:
            nbformat.write(notebook, file)

        print(f"Updated: {filename}")