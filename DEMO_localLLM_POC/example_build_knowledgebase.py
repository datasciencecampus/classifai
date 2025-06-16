from classifAI_API.utils.knowledgebase_factory import create_knowledgebase


create_knowledgebase(input_csv_data_filepath="./isco.csv",
                         csv_separator=",",
                         knowledgebase_output_dir="./", 
                         knowledgebase_filename="isco_knowledgebase_localLLM.parquet",
                         local_LLM="nomic-embed-text", 
                         all_local=True)

# create_knowledgebase(input_csv_data_filepath="./COICOP_processed.csv",
#                          csv_separator=",",
#                          knowledgebase_output_dir="./", 
#                          knowledgebase_filename="test_knowledgebase_genai.parquet",
#                          local_LLM=None, 
#                          all_local=False)
                         