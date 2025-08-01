from litellm import completion


def completion_prompt(prompt):
    response = completion(
        model="ollama/llama3.2:1b", 
        messages=[
            { 
                "role": "system",
                "content": "you are a helpful assistant that can answer questions about the context provided."
            },
            {
                "role": "user",
                "content": prompt,
            }
        ], 
        api_base="http://localhost:11434",
        stream=True
    )
    
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)


