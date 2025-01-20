import os
import google.generativeai as genai

genai.configure(api_key="AIzaSyD-lurOjjqQx_9YuhCObQVzpl9kWOGbQL8")

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
)

chat_session = model.start_chat(
  history=[
    {
      "role": "user",
      "parts": [
        "\"\"\"You are an advanced AI specialized in analyzing and summarizing structured data within tables. Your task is to generate a meaningful title and description for the provided table. The title should be concise and clearly reflect the table's primary content, while the description should summarize the context and provide insights into the key data. The description must be tailored for query ranking, meaning it should help identify the table's relevance to possible questions. Guidelines: \n1. Table Title:    \n    a.Provide a concise, clear title that reflects the content and context of the table.   \n    b.Be mindful of all shorthand terms in the table and expand them in the title and description. \n\n2.Table Description: \n    Summarize the main purpose of the table, the key columns, and the types of data it contains.   \n    a. Must include abbreviations or terms that might be unclear, expand them in the description.   \n    b. Identify and explain any patterns or notable trends. For example, if certain data points frequently appear together or if there are gaps, mention them.   \n    c. Point out the significance of key relationships in the data (e.g., correlations between columns or rows).   \n    d. If there are columns with categorical data (e.g., 'Region', 'Category'), mention their distribution and any specific categories of interest.  \n    e. Ensure the description answers potential questions about the data, such as 'What key information is this table providing?' or 'How can this table be used in answering specific questions?\nHere is the Table:\n\"['Season', 'North', 'South', 'East', 'West']['2000-01', 'EHC Regensburg', 'SB Rosenheim', 'Deggendorfer SC II', 'EV Landsberg']['2001-02', 'BTS Bayreuth', 'TuS Geretsried II', 'Deggendorfer SC II', 'TSV Kottern']['2002-03', 'EC Bayreuth', 'SC Riessersee II', 'TSV Erding Jets', 'TSV Peißenberg']['2003-04', 'EHC Ingolstadt II', 'ESC Holzkirchen', 'EHF Passau', 'TSV Kottern']['2004-05', 'VER Selb', 'EAC Bad Reichenhall', 'SE Freising', 'ERC Lechbruck']['2005-06', 'EC Bad Kissingen', 'EHC München II', 'ESV Waldkirchen', 'EV Bad Wörishofen']['2006-07', 'EHC Bayreuth', 'ESC Geretsried', 'ESV Gebensbach', 'VfL Denklingen']['2007-08', 'EC Amberg', 'SC Gaißach', 'EV Bruckberg', 'ESC Kempten']['2008-09', 'ERC Ingolstadt', 'TSV Schliersee', 'ESV Waldkirchen', '1. EC Senden']\n\"\n'\"\"\"\n\"\"\"You are an advanced AI specialized in analyzing and summarizing structured data within tables. Your task is to generate a meaningful title and description for the provided table. The title should be concise and clearly reflect the table's primary content, while the description should summarize the context and provide insights into the key data. The description must be tailored for query ranking, meaning it should help identify the table's relevance to possible questions. Guidelines: \n1. Table Title:    \n    a.Provide a concise, clear title that reflects the content and context of the table.   \n    b.Be mindful of all shorthand terms in the table and expand them in the title and description. \n\n2.Table Description: \n    Summarize the main purpose of the table, the key columns, and the types of data it contains.   \n    a. Must include abbreviations or terms that might be unclear, expand them in the description.   \n    b. Identify and explain any patterns or notable trends. For example, if certain data points frequently appear together or if there are gaps, mention them.   \n    c. Point out the significance of key relationships in the data (e.g., correlations between columns or rows).   \n    d. If there are columns with categorical data (e.g., 'Region', 'Category'), mention their distribution and any specific categories of interest.  \n    e. Ensure the description answers potential questions about the data, such as 'What key information is this table providing?' or 'How can this table be used in answering specific questions?\nHere is the Table:\n\"['Season', 'North', 'South', 'East', 'West']['2000-01', 'EHC Regensburg', 'SB Rosenheim', 'Deggendorfer SC II', 'EV Landsberg']['2001-02', 'BTS Bayreuth', 'TuS Geretsried II', 'Deggendorfer SC II', 'TSV Kottern']['2002-03', 'EC Bayreuth', 'SC Riessersee II', 'TSV Erding Jets', 'TSV Peißenberg']['2003-04', 'EHC Ingolstadt II', 'ESC Holzkirchen', 'EHF Passau', 'TSV Kottern']['2004-05', 'VER Selb', 'EAC Bad Reichenhall', 'SE Freising', 'ERC Lechbruck']['2005-06', 'EC Bad Kissingen', 'EHC München II', 'ESV Waldkirchen', 'EV Bad Wörishofen']['2006-07', 'EHC Bayreuth', 'ESC Geretsried', 'ESV Gebensbach', 'VfL Denklingen']['2007-08', 'EC Amberg', 'SC Gaißach', 'EV Bruckberg', 'ESC Kempten']['2008-09', 'ERC Ingolstadt', 'TSV Schliersee', 'ESV Waldkirchen', '1. EC Senden']\n\"\n'\"\"\"\n",
      ],
    },
    {
      "role": "model",
      "parts": [
        "**Table Title:** German Ice Hockey League Teams by Region (2000-2009)\n\n**Table Description:**\n\nThis table displays the participating teams in a German ice hockey league, categorized by geographical region, across ten seasons from 2000-2001 to 2008-2009.  The columns represent the hockey season (Season) and the four regional divisions: North, South, East, and West.  Each cell contains the name of a team that competed in that specific region during that season.  Note that many team names include abbreviations, for example, \"EHC\" likely stands for \"Eishockey-Club\" (Ice Hockey Club), \"SC\" for \"Sport-Club\" (Sports Club), \"TSV\" for \"Turn- und Sportverein\" (Gymnastics and Sports Club), and \"EV\" for \"Eishockeyverein\" (Ice Hockey Association).\n\nThe data reveals the dynamic nature of team participation, with frequent changes in team membership across seasons and regions.  There is no clear pattern of dominance from any particular team or region throughout the ten-year period. Some teams appear multiple times, indicating consistent participation or possibly relegations and promotions. For example, TSV Kottern and ESV Waldkirchen appear in multiple seasons.  This table can be used to answer questions such as: \"Which teams played in the North region during the 2005-2006 season?\", \"How many seasons did a specific team participate?\",  \"What was the regional distribution of teams across the league during a given season?\" and \"Were there any teams that consistently competed in the same region?\"  The table allows for analysis of team longevity and regional competitiveness over the specified period.\n",
      ],
    },
  ]
)

response = chat_session.send_message("Hi")

print(response.text)