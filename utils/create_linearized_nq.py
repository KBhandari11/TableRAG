import json
import sys

# collections_dict = {}
def convert_table_to_token_format(table_json, table_index):
    """
    Convert a table JSON object into a single string with special tokens.
    
    Format:
      {table_index}\t<SOT> {caption} <EOT> <BOC> {header_cell1} <SOC> {header_cell2} ... <EOC> 
      <BOR>{row1_cell1} <SOR> {row1_cell2} <SOR> {row1_cell3} <EOR> <BOR>{row2_cell1} ... <EOR> ...
    
    The caption is taken from "documentTitle" if available.
    """
    # Get caption (if available)
    caption = table_json.get("documentTitle", "").strip()
    givenTableID = table_json.get("tableId", "").strip()
    caption_token = f"<SOT> {caption} <EOT>"
    
    # Build header using the "columns" list.
    columns = table_json.get("columns", [])
    header_cells = [col.get("text", "").strip() for col in columns]
    if header_cells:
        # First header cell gets the <BOC> token.
        header_token = "<BOC> " + header_cells[0]
        # Subsequent header cells get the <SOC> token.
        for cell in header_cells[1:]:
            header_token += " <SOC> " + cell
        header_token += " <EOC>"
    else:
        header_token = ""
    
    # Build the body rows using the "rows" list.
    row_tokens = []
    for row in table_json.get("rows", []):
        cells = row.get("cells", [])
        cell_texts = [cell.get("text", "").strip() for cell in cells]
        if not cell_texts:
            continue
        # First cell: prepend <BOR>
        row_str = "<BOR>" + cell_texts[0]
        # Next cells: prepend <SOR>
        for cell_text in cell_texts[1:]:
            row_str += " <SOR> " + cell_text
        row_str += " <EOR>"
        row_tokens.append(row_str)
    
    # Join all row tokens with a space.
    rows_token_str = " ".join(row_tokens)
    
    return caption_token,header_token, rows_token_str, givenTableID

def create_linearized_nq(input_file, output_file):
   

    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        fout.write(f"index,table_text\n")
        for idx, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                table_json = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON: {line}")
                continue
            
            caption_token,header_token, rows_token_str, givenTableID = convert_table_to_token_format(table_json, idx)
            caption_token = caption_token.replace('\t', '')
            header_token = header_token.replace('\t', '')
            rows_token_str = rows_token_str.replace('\t', '')
            givenTableID = givenTableID.replace('\t', '')

            # collections_dict[str(idx-1)] = f"{caption_token} {header_token} {rows_token_str}"
            fout.write(f"{idx-1},\"{caption_token} {header_token} {rows_token_str}\"\n")

if __name__ == "__main__":
    input_file, output_file = sys.argv[1], sys.argv[2]
    create_linearized_nq(input_file, output_file)