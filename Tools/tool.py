# If we need to integrate our won api, want to perform action using our data, and want to write you own logic then we need to create custom tools 
from langchain.tools import tool

# adding  tool decorator/annotation 
# Adding doc string 
#  adding type hints

@tool("Add") # added tool decorator/annotation
def sum(a:int, b:int) -> int:  # here added type hints
    """Addition of two numbers"""  # here added doc strings
    return a+b

result=sum.invoke({"a":1,"b":2})
print(result)

print("name of the tool: ",sum.name)
print("Description of the tool: ",sum.description)
print("Argument passed to the tool: ",sum.args)
print("Schema of about our tool sending to LLM: ", sum.args_schema.model_json_schema) # to see schema of out tool