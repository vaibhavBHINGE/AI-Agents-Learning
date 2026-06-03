from langchain_core.prompts import  PromptTemplate

prompt=PromptTemplate(
    input_variables=["team_name","language"],
    validate_template=True,
    template="Generate the calculations for about the IPL team {team_name}. "
             "Give me the percentage chance of that team qualifying for the playoffs compared to other teams. "
             "Also analyze {team_name}'s remaining matches and their win probability (strengths/weaknesses) for those matches given details should be more foucsed on mathametical or more statustical analysis in "
             "{language} language. one thing is dont dont write as report or essay "
)
prompt.save("prompt_template.json")