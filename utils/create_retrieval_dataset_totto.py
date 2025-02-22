import random
import pandas as pd
import json
import copy

# Using json.JSONEncoder for customization
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, tuple):
            return list(obj)
        return super().default(obj)
    
def table_to_text(table):
    """
    Converts a table (list of lists) to a string.
    Each row is concatenated with spaces, and rows are joined with newlines.
    """
    return " ".join([" ".join(map(str, row)) for row in table])


def create_table_retrieval_dataset(list_tables, data, num_table_retrieval=50):
    '''
    Creates a new dataset that selects random other num_table_retrieval number of table information for each datapoint. 
    Objective: Use the "summary" to extract the relevant table among the list of num_table_retrieval tables 
    '''
    new_dataset = []
    for idx, d in enumerate(data):
        new_data_point = {}
        sampled_keys = random.sample([key for key in list_tables.keys() if key != idx], num_table_retrieval-1)
        sampled_keys.append(idx)
        random.shuffle(sampled_keys) 
        new_data_point["index"] = idx
        new_data_point["actual_table_info"] = list_tables[idx]   
        new_data_point["random_retrieval_sample"] = (idx, [key for key in sampled_keys])
        #new_data_point["list_title_tab-description_retrieval"] = (list_tables[idx]["title_table-description"], [list_tables[key]["title_table-description"] for key in sampled_keys])
        #new_data_point["list_title_column_header_retrieval"] = (list_tables[idx]["title_col_info"],[list_tables[key]["title_col_info"] for key in sampled_keys])
        #new_data_point["list_title_col_table_retrieval"] = (list_tables[idx]["title_col_table_text"],[list_tables[key]["title_col_table_text"] for key in sampled_keys])
        #new_data_point["list_exact_row_retrieval"] = (list_tables[idx]["title_col_exact"],[list_tables[key]["title_col_exact"] for key in sampled_keys])
        new_data_point["summary"] = list_tables[idx]["summary"] 
        new_dataset.append(new_data_point)
    return new_dataset

def getRowColumn(table: list)-> tuple:
    '''
    get the maximum row and column size by computing the col and row span
    '''
    rowHeight = 0
    colWidth = 0
    ctr = 0##
    for row in table:
        
        #print(ctr, row, "\n")##
        if row != []:
            rowHeight += int(row[0]['row_span'])
        else:
            rowHeight += 1
        c = 0
        for col in row:
            c+= int(col['column_span'])
        colWidth = max(colWidth, c)
        ctr+=1##
    return rowHeight, colWidth

def convert2by2(table, highlighted)->  dict:
    '''
    Convert the given table into 2x2 table. Adjusting to row and column span. 
    If the given table has some rowspan and columnspan issue, it might through out an error. 
    '''
    #cell keys => ['value', 'is_header', 'column_span', 'row_span']
    rowIDX, colIDX =  getRowColumn(table)
    #print(rowIDX, colIDX)
    newTable = [[{"span":False} for _ in range(colIDX)] for _ in range(rowIDX)]
    newHighlight = []
    #print(0)
    for r, row in enumerate(table):
        skip = 0
        
        #print("\nrow: ", r, row,"\n")
        for c, cell in enumerate(row):

            #print(c, skip,"\npre new table: ", newTable[r])
            
            if all('value' in col_dict for col_dict in newTable[r]):
               continue
            
            while newTable[r][c+skip]['span']:
                skip +=1
            if [r,c] in highlighted:
                newHighlight.append([r,c+skip])
            newTable[r][c+skip] = copy.deepcopy(cell)
            newTable[r][c+skip]['span'] = False
            if int(cell["row_span"]) != 1 and int(cell["row_span"]) != 99:
                    for i in range(1,int(cell["row_span"])):
                        if r+i < rowIDX:
                            newTable[r+i][c+skip] = copy.deepcopy(cell)
                            newTable[r+i][c+skip]['span'] = True
            if int(cell["column_span"]) != 1:
                for i in range(1,int(cell["column_span"])):
                    if c+i < len(row):
                        newTable[r][c+skip+i] = copy.deepcopy(cell)
                        newTable[r][c+skip+i]['span'] = True
                        #skip+=1
            #print(c, skip,"\npost new table size: ", len(newTable[r]),newTable[r])
    
    # pop {span: false}
    while not any('value' in col_dict for col_dict in newTable[-1]):
        newTable.pop()
    
    for rowidx, row in enumerate(newTable):
        for colidx, cell in enumerate(row):
            if not "value" in cell:
                if "value" in row[colidx-1]:
                   cell["value"] =  copy.deepcopy(row[colidx-1]["value"])
                else:
                    cell["value"] = ""
    return newTable, newHighlight

def convertTable(tableDict: dict,highLight: list)-> list:
    '''
    Convert table into ready to display format table
    '''
    noRow, noCol =  len(tableDict),len(tableDict[0]) 
    table = [["" for c in range(noCol)] for r in range(noRow)]
    r_i = 0
    for r, row in enumerate(tableDict):
        c_i = 0
        for c, col in enumerate(row):
            if r_i >= noRow or c_i >= noCol:
                break
            while (table[r_i][c_i]) != "":
                c_i += 1
            if [r,c] in highLight:
                table[r_i][c_i] = '\033'+str(col['value'])+'\033'
            else:
                table[r_i][c_i] = str(col['value'])
        r_i += 1
    return table

def getDashes(inputTable):
    maxLength = [len(str(max(i, key=lambda x: len(str(x)))))for i in  zip(*inputTable)]
    dashes = ["".join(["-" for l in range(lengths) ]) for lengths in maxLength]   
    return '|'.join(dashes), maxLength

def displayTable(exampleTable, highlightedCell):
    inputTable = convertTable(exampleTable, highlightedCell)
    #inputTable =formatTable(exampleTable)
    dashes, maxLen =  getDashes(inputTable)
    print(dashes)
    for row in inputTable:
        print('|'.join('{0:{width}}'.format(x, width=y+11) if not(x.find('\033[1;37m')) else '{0:{width}}'.format(x, width=y) for x, y in zip(row, maxLen)), end='|')
        print("\n"+dashes)
    return inputTable

def get_exact_column_header(table, highlighted_cell):
    newTable, newHighlight = convert2by2(table, highlighted_cell) 
    highlighted_row, highlighted_col = set(), set() 
    for [rowidx, colidx] in newHighlight:
        highlighted_row.add(rowidx)
        highlighted_col.add(colidx)
    exact_col_header = ""
    for rowidx, row in enumerate(newTable):
        for colidx, cell in enumerate(row):
            if "is_header" in cell: 
                if (rowidx in highlighted_row or colidx in highlighted_col) and cell["is_header"]:
                    exact_col_header += f"{cell['value']} "
    return exact_col_header 
    
def get_table_list(data):
    table_list = {}
    for idx, d in enumerate(data):
        table = [] 
        header_info = ""
        exact_row_info = ""
        for rowidx, row in enumerate(d["table"]):
            for colidx, cell in enumerate(row):
                table.append([cell['value']]) 
                if cell["is_header"]:
                    header_info += f"{cell['value']} "
                if [rowidx, colidx] in d["highlighted_cells"]:
                    exact_row_info += f"{cell['value']} " 

        table_list[idx] = {"index":idx,
                           "summary": d["sentence_annotations"][0]["final_sentence"],
                           "title_table-description": f"{d['table_page_title']} {d['table_section_title']} {d['table_section_text']}",
                           "column_header_info": header_info,
                           "title_col_info":f"{d['table_page_title']} {d['table_section_title']} {d['table_section_text']} {header_info}",
                           "table_text": table_to_text(table),
                           "title_col_table_text": f"{d['table_page_title']} {d['table_section_title']} {d['table_section_text']} {table_to_text(table)}", 
                           "exact_row_information": exact_row_info, 
                           "title_col_exact": f"{d['table_page_title']} {d['table_section_title']} {d['table_section_text']} {get_exact_column_header(d['table'], d['highlighted_cells'])} {exact_row_info}", 
                           "highlighted_cells":d["highlighted_cells"] 
                           } 
    return table_list

if __name__ == "__main__":
    totto_dataset_path = "/gpfs/u/home/LLMG/LLMGbhnd/scratch/tableRAG_data/totto/totto_data/totto_dev_data.jsonl" 
    
    data = []
    with open(totto_dataset_path, 'r') as file:
        for line_number,line in enumerate(file):
            data.append(json.loads(line))

    table_list = get_table_list(data) 
    # save the table_list as a csv file to index the data
    table_df = pd.DataFrame([table_list[idx] for idx in table_list])
    table_df.to_csv('/gpfs/u/home/LLMG/LLMGbhnd/scratch/tableRAG_data/totto/table_list.csv', index=False)

    for n_val in [2000,"all"]: #[50, 100, 200, 500, 
        if n_val == "all":
            n = len(data)
        else:
            n = n_val
        new_dataset = create_table_retrieval_dataset(table_list, data, num_table_retrieval=n)
        with open(f"/gpfs/u/home/LLMG/LLMGbhnd/scratch/tableRAG_data/totto/retrieval/totto_retrieval_{n_val}.json", 'w') as f:
            json.dump(new_dataset, f, cls=CustomEncoder, indent=4)  # `indent=4` for pretty formatting
        print(f"Data successfully saved to data/retrieval_data/totto_retrieval_{n_val}.json",flush=True)