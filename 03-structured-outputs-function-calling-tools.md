# Part 3: Structured Outputs, Function Calling & Native Tools

This section covers three powerful capabilities of the Gemini API: structured outputs for extracting information into defined schemas, function calling for connecting to external tools and APIs, and native tools like Google Search for enhanced capabilities.


```python
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List, Optional
import sys
import os
from IPython.display import Image, Markdown

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    from google.colab import userdata
    GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')
else:
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY',None)

# Create client with api key
MODEL_ID = "gemini-2.5-flash-preview-05-20"
client = genai.Client(api_key=GEMINI_API_KEY)
```

## 1. Structured Outputs

Structured outputs allow you to constrain Gemini to respond with JSON in a specific format instead of unstructured text. This is essential for:
- **Data extraction**: Converting unstructured text into structured data
- **API integration**: Getting consistent formats for downstream processing  
- **Database insertion**: Ensuring data matches your schema requirements
- **Quality control**: Validating that responses contain required fields


```python
class Recipe(BaseModel):
    recipe_name: str
    ingredients: List[str]
    prep_time_minutes: int
    difficulty: str  # "easy", "medium", "hard"
    servings: int

class RecipeList(BaseModel):
    recipes: List[Recipe]

# Using Pydantic models for structured output
response = client.models.generate_content(
    model=MODEL_ID,
    contents="Give me 2 popular cookie recipes with ingredients and prep details.",
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=RecipeList,
    ),
)

# Get structured data directly
recipes: List[Recipe] = response.parsed
for recipe in recipes.recipes:
    print(f"Recipe: {recipe.recipe_name}")
    print(f"Ingredients: {recipe.ingredients}")
    print(f"Prep Time: {recipe.prep_time_minutes} minutes")
    print(f"Difficulty: {recipe.difficulty}")
    print(f"Servings: {recipe.servings}")
    print("\n")
```

    Recipe: Classic Chocolate Chip Cookies
    Ingredients: ['all-purpose flour', 'baking soda', 'salt', 'unsalted butter', 'granulated sugar', 'brown sugar', 'eggs', 'vanilla extract', 'chocolate chips']
    Prep Time: 20 minutes
    Difficulty: Easy
    Servings: 24
    
    
    Recipe: Chewy Oatmeal Raisin Cookies
    Ingredients: ['all-purpose flour', 'baking soda', 'cinnamon', 'salt', 'unsalted butter', 'brown sugar', 'granulated sugar', 'eggs', 'vanilla extract', 'rolled oats', 'raisins']
    Prep Time: 15 minutes
    Difficulty: Medium
    Servings: 36
    
    


## !! Exercise: PDF to Structured Data !!

Extract structured information from a PDF invoice or document using the Files API and structured outputs.

Tasks:
- A Pydantic schema (`InvoiceItem` and `InvoiceData`) and an example PDF file path (`../assets/data/rewe_invoice.pdf`) are provided. You can use this or replace it with your own PDF invoice.
- Upload the PDF file using `client.files.upload()`.
- Call `client.models.generate_content()`:
- Access the parsed structured data from `response.parsed`.



```python
class InvoiceItem(BaseModel):
    description: str
    quantity: int
    unit_price: float
    total: float

class InvoiceData(BaseModel):
    invoice_number: str
    date: str
    vendor_name: str
    vendor_address: str
    total_amount: float
    items: List[InvoiceItem]

# Upload a PDF file (replace with your PDF path)
pdf_file_path = "../assets/data/rewe_invoice.pdf"

# Upload the file
file_id = client.files.upload(file=pdf_file_path)

# Extract structured data from PDF
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        "Extract all invoice information from this PDF including items, vendor details, and totals.",
        file_id
    ],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=InvoiceData,
    ),
)

# Get structured invoice data
invoice: InvoiceData = response.parsed
print(f"Invoice #{invoice.invoice_number}")
print(f"Vendor: {invoice.vendor_name}")
print(f"Date: {invoice.date}")
print(f"Total: ${invoice.total_amount}")
print(f"Items ({len(invoice.items)}):")
for item in invoice.items:
    print(f"  - {item.description}: {item.quantity} x ${item.unit_price} = ${item.total}")
```

    Invoice #7077
    Vendor: REWE Marc Balkow OHG
    Date: 2025-03-29
    Total: $22.67
    Items (10):
      - BUTTERKAESE: 1 x $1.79 = $1.79
      - BRIOCHE BUNS 4ER: 1 x $1.69 = $1.69
      - HUMMUS NATUR: 1 x $0.99 = $0.99
      - SENSATIONAL BUR.: 1 x $3.19 = $3.19
      - BANANE CHIQUITA: 0 x $2.39 = $0.7
      - SUESSKARTOFFEL: 0 x $4.49 = $1.99
      - EISBERGSALAT: 1 x $1.19 = $1.19
      - CHERRYROMATOMATE: 1 x $1.15 = $1.15
      - BIO EIER KL. S-L: 1 x $2.99 = $2.99
      - PISTAZIENCREME: 1 x $6.99 = $6.99


## 2. Function Calling

Function calling allows Gemini to intelligently decide when to call specific functions you define. This enables:
- **External API integration**: Connect to weather, stocks, databases
- **Dynamic calculations**: Perform real-time computations
- **System interaction**: Execute commands or retrieve system information
- **Multi-step workflows**: Chain function calls for complex tasks


```python

def get_weather(location: str) -> dict:
    """Gets current weather for a location.
    
    Args:
        location: The city name, e.g. "San Francisco"
        
    Returns:
        Weather information dictionary
    """
    # Mock weather data - in real use, you'd call a weather API
    weather_data = {
        "temperature": 22,
        "condition": "sunny", 
        "humidity": 60,
        "location": location,
        "feels_like": 24
    }
    print(f"🌤️ FUNCTION CALLED: get_weather(location='{location}')")
    return weather_data

# Define function declarations for the model
weather_function = {
    "name": "get_weather",
    "description": "Gets current weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city name"
            }
        },
        "required": ["location"]
    }
}
tools = [types.Tool(function_declarations=[weather_function])]

# Define user prompt
contents = [
    types.Content(
        role="user", parts=[types.Part(text="Whats the weather in Tokyo?")]
    )
]


# Send request with function declarations
response = client.models.generate_content(
    model=MODEL_ID,
    contents=contents,
    config=types.GenerateContentConfig(tools=tools)
)

# Check for function calls
function_call = response.candidates[0].content.parts[0].function_call
print(f"Model wants to call: {function_call.name}")
print(f"With arguments: {dict(function_call.args)}")

# Execute the function
if function_call.name == "get_weather":
    result = get_weather(**function_call.args)
else:
    result = {"error": "Unknown function"}

print(f"Function result: {result}")

# Send function result back to model
function_response_part = types.Part.from_function_response(
    name=function_call.name,
    response={"result": result}
)
# Append function call and result of the function execution to contents
contents.append(types.Content(role="model", parts=[types.Part(function_call=function_call)])) # Append the model's function call message
contents.append(types.Content(role="user", parts=[function_response_part])) # Append the function response

# Get final response
final_response = client.models.generate_content(
    model=MODEL_ID,
    contents=contents,
    config=types.GenerateContentConfig(tools=tools)
)

print(f"\nFinal response: {final_response.text}")
```

    Model wants to call: get_weather
    With arguments: {'location': 'Tokyo'}
    🌤️ FUNCTION CALLED: get_weather(location='Tokyo')
    Function result: {'temperature': 22, 'condition': 'sunny', 'humidity': 60, 'location': 'Tokyo', 'feels_like': 24}
    
    Final response: The weather in Tokyo is sunny with a temperature of 22 degrees Celsius, and it feels like 24 degrees. The humidity is 60%.


### Automatic Function Calling (Python Only)

The Python SDK can automatically handle function execution for you:


```python
def calculate_area(length: float, width: float) -> dict:
    """Calculate the area of a rectangle.
    
    Args:
        length: Length of the rectangle
        width: Width of the rectangle

    Returns:
        Price calculations
    """
    area = length * width
    print(f"CALC: {length} × {width} = {area}")
    return {"operation": "area", "result": area}

# Using automatic function calling - much simpler!
config = types.GenerateContentConfig(
    tools=[get_weather, calculate_area]  # Pass functions directly
)

response = client.models.generate_content(
    model=MODEL_ID,
    contents="What's the weather in Tokyo and what's the area of a 5x3 meter room?",
    config=config
)

print(response.text)  # SDK handles function calls automatically
```

    🌤️ FUNCTION CALLED: get_weather(location='Tokyo')
    CALC: 5 × 3 = 15
    The weather in Tokyo is sunny with a temperature of 22 degrees Celsius, and it feels like 24 degrees Celsius. The humidity is 60%. The area of a 5x3 meter room is 15 square meters.


## !! Exercise: Calculator Agent !!

Create a set of calculator functions (add, subtract, multiply, divide) and use Gemini's function calling capability to perform calculations based on natural language prompts.

Tasks:
- Define Python functions for `add(a: float, b: float)`, `subtract(a: float, b: float)`, `multiply(a: float, b: float)`, and `divide(a: float, b: float)`.
- Create a list of these functions (`calculator_tools`).
- Test single operations with `client.models.generate_content()`:
- Test complex expressions with a prompt for a multi-step calculation (e.g., "Calculate (25 + 15) * 3 - 10. Do this step by step.").


```python
def add(a: float, b: float) -> dict:
    """Add two numbers.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Sum of the numbers
    """
    result = a + b
    print(f"CALC: {a} + {b} = {result}")
    return {"operation": "addition", "result": result}

def subtract(a: float, b: float) -> dict:
    """Subtract two numbers.
    
    Args:
        a: First number  
        b: Second number
        
    Returns:
        Difference of the numbers
    """
    result = a - b
    print(f"CALC: {a} - {b} = {result}")
    return {"operation": "subtraction", "result": result}

def multiply(a: float, b: float) -> dict:
    """Multiply two numbers.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Product of the numbers
    """
    result = a * b
    print(f"CALC: {a} × {b} = {result}")
    return {"operation": "multiplication", "result": result}

def divide(a: float, b: float) -> dict:
    """Divide two numbers.
    
    Args:
        a: Dividend
        b: Divisor
        
    Returns:
        Quotient of the numbers
    """
    if b == 0:
        print(f"CALC: Error - Division by zero")
        return {"operation": "division", "error": "Division by zero"}
    
    result = a / b
    print(f"CALC: {a} ÷ {b} = {result}")
    return {"operation": "division", "result": result}

# Test the calculator with automatic function calling
calculator_tools = [add, subtract, multiply, divide]

# Single operation
response = client.models.generate_content(
    model=MODEL_ID,
    contents="What is 15 multiplied by 7?",
    config=types.GenerateContentConfig(tools=calculator_tools)
)
print("Single operation:")
print(response.text)

# Complex calculation
response = client.models.generate_content(
    model=MODEL_ID,
    contents="Calculate (25 + 15) x 3 - 10. Do this step by step.",
    config=types.GenerateContentConfig(tools=calculator_tools)
)
print("\nComplex calculation:")
print(response.text)
```

    CALC: 15 × 7 = 105
    Single operation:
    15 multiplied by 7 is 105.
    CALC: 25 + 15 = 40
    CALC: 40 × 3 = 120
    CALC: 120 - 10 = 110
    
    Complex calculation:
    Finally, 120 - 10 = 110.
    
    So, (25 + 15) x 3 - 10 = 110.


## 3. Native Tools

Gemini provides native tools for enhanced capabilities like searching the web and analyzing URL content.

### Google Search Integration

**Use cases:**
- Current events and news
- Real-time data lookup
- Fact verification
- Research assistance


```python
# Define Google Search tool
google_search_tool = types.Tool(google_search=types.GoogleSearch())

# Current events query
response = client.models.generate_content(
    model=MODEL_ID,
    contents="What are the latest developments in renewable energy technology in 2025?",
    config=types.GenerateContentConfig(
        tools=[google_search_tool],
    )
)

print("🔍 Current Renewable Energy News:")
print(response.text)
```

    🔍 Current Renewable Energy News:
    In 2025, the renewable energy sector is experiencing significant advancements across various technologies, driven by increasing investment, policy support, and the urgent need for sustainable solutions. Renewable electricity is projected to surpass coal as the world's leading energy source.
    
    Key developments include:
    
    **Solar Power:**
    *   **Perovskite Solar Cells** are a major innovation, offering higher efficiency (potentially over 30%) at a lower production cost compared to traditional silicon panels. Challenges remain in stability and scaling production, but advancements in protective encapsulation layers and materials are underway.
    *   **Bifacial Solar Panels** are gaining traction, capable of capturing sunlight from both sides, increasing efficiency and energy output, especially in reflective environments like snow or sand. Their costs are falling, making them more accessible for commercial and residential use.
    *   **Solar Canopies** are being developed for electric vehicle (EV) chargers, providing a consistent source of clean electricity. Some initiatives are utilizing bifacial panels for these canopies.
    *   **Floating Solar Farms (Floatovoltaics)** are expanding as a solution to land scarcity, with the cooling effect of water enhancing their efficiency. Asia is leading in adoption.
    *   **Transparent Solar Panels** are making it possible to integrate solar energy into everyday surfaces like windows and facades, turning buildings into energy generators.
    *   **Solar Paint and Coatings** are under development, containing photovoltaic materials that allow building exteriors to generate electricity.
    *   **Anti-solar panels** are being explored to generate power at night by capturing radiated heat from the Earth in the form of infrared radiation.
    *   **Photovoltaic Noise Barriers (PVNB)** are being integrated with PV modules along roadways to generate electricity from solar energy.
    *   **Smart Inverters** are becoming more prevalent, reacting to grid patterns to manage load, prevent blackouts, and make solar panels more responsive.
    *   **AI** is enhancing solar power generation, storage, and distribution through optimization.
    
    **Wind Energy:**
    *   **Advanced Turbine Technology** focuses on larger and more efficient wind turbines, with increased power generation capacity per unit to lower costs. This includes innovations in blade aerodynamics and materials.
    *   **Residential Wind Turbines**, including bladeless designs, are being developed for households and microgrids, addressing land use concerns and diversifying renewable electricity sources.
    *   **Offshore Wind Energy** is expanding, with floating turbines enabling installations in deeper waters and harnessing stronger, more consistent winds. This segment is expected to see significant growth.
    *   **Vertical Axis Wind Turbines (VAWTs)** are gaining importance for urban environments or areas with variable wind patterns due to their ability to capture wind from any direction.
    *   **Wooden Turbine Towers** are being explored to reduce production costs and emissions compared to steel.
    *   **Digitalization and AI-based systems** are optimizing wind farm operations and maintenance by predicting failures and enhancing turbine efficiency through advanced data analytics and sensor technology.
    
    **Energy Storage:**
    *   **Advanced Battery Storage Solutions** are crucial for balancing renewable energy supply and demand. The U.S. is projected to add a record 18.2 gigawatts of battery storage capacity in 2025.
    *   **Lithium-ion batteries** continue to dominate, but research is focused on enhanced versions like Li-Polymer, Li-Air, and Li-Sulfur for increased efficiency and safety.
    *   **Lithium Alternatives** are gaining traction due to sustainability concerns, including sodium-ion, zinc-air, nickel-zinc, and magnesium-ion batteries, offering advantages in cost, safety, and abundance of materials. China has launched a large sodium-ion Battery Energy Storage System (BESS).
    *   **Solid-state, flow, and thermal batteries** are surpassing lithium-ion with higher energy density, longer life, and greater safety.
    *   **Sand Batteries**, a novel form of storage residing in fluidized beds of silica sand, have been revealed by Italian energy experts.
    *   **Electric Vehicles (EVs) as storage** are being explored, allowing vehicles to act as energy storage units.
    *   **Long-duration energy storage systems** (exceeding eight hours of storage) are anticipated to more than double in installations compared to 2024, essential for grid stability with intermittent renewables.
    *   **Battery recycling technologies** are advancing to extract valuable materials and promote a circular economy for batteries.
    *   **Potassium-ion batteries** are being developed as a cheaper, longer-lasting alternative for large-scale energy storage due to potassium's abundance.
    *   **Quasi-solid-state lithium-ion batteries** are being developed to address flammability and short lifespan issues of traditional lithium-ion cells, showing improved thermal stability and cycle performance.
    
    **Other Emerging Technologies and Trends:**
    *   **Green Hydrogen** is emerging as a zero-carbon fuel for difficult-to-electrify sectors like heavy industry and long-haul transportation. Produced through water electrolysis powered by renewable energy, it provides a clean alternative for decarbonization. Investments and projects in green hydrogen are growing significantly.
    *   **Renewable Natural Gas** involves capturing methane produced by cows, which would otherwise contribute to greenhouse gas emissions, and transforming it into a power source.
    *   **Artificial Intelligence (AI) and Data Analytics** are playing critical roles in optimizing energy systems, including predictive grid and facility maintenance, load forecasting, market optimization, and enhancing solar and wind farm efficiency. AI's increasing energy demand is also driving the need for rapid deployment of clean and reliable energy sources.
    *   **Distributed Energy Resources (DERs) & Microgrids** are growing, with rooftop solar, small wind, and local battery systems improving energy reliability and reducing strain on electrical transmission lines.
    *   **Phase-Change Materials for Insulation** are enhancing building elements to better harness warmth, for example, in solar chimneys.
    *   **Food Waste-Powered Panels** are an emerging innovation in alternative energy.
    *   **Carbon Capture and Storage (CCS)** is another area of focus within clean energy technology.
    *   **Blockchain in energy management** is also an area of innovation.


### URL Context Tool

**Use cases:**
- Website content analysis
- Documentation summarization
- Competitive research
- Content extraction


```python
# URL context for analyzing specific web pages
url_context_tool = types.Tool(url_context=types.UrlContext())

response = client.models.generate_content(
    model=MODEL_ID,
    contents="Summarize the key features and benefits mentioned on https://www.python.org/about/ in 3 bullet points.",
    config=types.GenerateContentConfig(
        tools=[url_context_tool],
    )
)

print("🌐 Python.org Summary:")
print(response.text)
```

    🌐 Python.org Summary:
    Python is highlighted for its power, speed, and cross-platform compatibility, allowing it to run everywhere. It is also emphasized as friendly and easy to learn, appealing to both new and experienced programmers. Furthermore, Python's open-source nature, administered by the Python Software Foundation, makes it freely usable and distributable, even for commercial purposes.


### Code Execution Tool

Gemini can execute Python code to perform calculations, create visualizations, and process data.


```python
# Code execution tool
code_execution_tool = types.Tool(code_execution={})

response = client.models.generate_content(
    model=MODEL_ID,
    contents="Create a bar chart showing the population of the 5 largest cities in the world. Use matplotlib.",
    config=types.GenerateContentConfig(
        tools=[code_execution_tool],
    )
)


for p in response.candidates[0].content.parts:
    if p.text:
        display(Markdown(p.text))
    elif p.executable_code:
        display(Markdown(f"```python\n{p.executable_code.code}\n```"))
    elif p.inline_data:
        display(Image(data=p.inline_data.data, width=800, format="png"))

```


To create a bar chart showing the population of the 5 largest cities in the world, I first need to gather the relevant data: the names of the cities and their estimated populations. It's important to note that city population figures can vary depending on the definition used (e.g., city proper, urban area, metropolitan area) and the source/year of estimation.

For this example, I will use estimated populations for the urban areas of the following five cities, based on common recent data (e.g., from sources like World Population Review, 2023/2024 estimates).

**Cities and their Estimated Urban Area Populations (in millions):**
1.  **Tokyo, Japan:** ~37.2 million
2.  **Delhi, India:** ~32.9 million
3.  **Shanghai, China:** ~29.2 million
4.  **Dhaka, Bangladesh:** ~23.2 million
5.  **São Paulo, Brazil:** ~22.6 million

Now, I will use `matplotlib` to create the bar chart.





```python
import matplotlib.pyplot as plt
import numpy as np

# Data for the 5 largest cities (urban area population estimates)
cities = ['Tokyo', 'Delhi', 'Shanghai', 'Dhaka', 'São Paulo']
populations = [37.2, 32.9, 29.2, 23.2, 22.6] # in millions

# Create the bar chart
plt.figure(figsize=(10, 6)) # Set the figure size for better readability
plt.bar(cities, populations, color='skyblue')

# Add labels and title
plt.xlabel('City')
plt.ylabel('Population (Millions)')
plt.title('Top 5 Largest Cities by Urban Area Population (Estimated)')
plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for better fit
plt.grid(axis='y', linestyle='--', alpha=0.7) # Add a grid for better readability
plt.tight_layout() # Adjust layout to prevent labels from being cut off

# Display the plot
plt.show()

```



    
![png](03-structured-outputs-function-calling-tools_files/03-structured-outputs-function-calling-tools_17_2.png)
    



The image you provided, `output_2025-05-30-23-55-48-263270.png`, successfully displays the bar chart showing the estimated urban area populations of the 5 largest cities: Tokyo, Delhi, Shanghai, Dhaka, and São Paulo.

The chart is clear and correctly depicts the data as requested in the previous prompt, with appropriate labels, title, and formatting.


## !! Exercise: Data Analysis with Code Execution !!

Combine Google Search and Code Execution tools to find real-world data and then analyze or visualize it using Python code generated and executed by Gemini.

Tasks:
- Define a prompt that requires both searching for information and then processing/visualizing that information. For example: "Search for the population of the 5 largest cities in the world and create a bar chart of their populations."
- Create a `types.Tool` for code execution: `code_execution_tool = types.Tool(code_execution={})`.
- Create a `types.Tool` for Google Search: `google_search_tool = types.Tool(google_search=types.GoogleSearch())`.
- Call `client.models.generate_content()`:
- Iterate through the `response.candidates[0].content.parts` and display the parts.


```python
prompt = """
Search for the population of the 5 largest companies in the world and create a bar chart.
"""

code_execution_tool = types.Tool(code_execution={})
google_search_tool = types.Tool(google_search=types.GoogleSearch())

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config=types.GenerateContentConfig(
        tools=[code_execution_tool, google_search_tool],
    )
)

for p in response.candidates[0].content.parts:
    if p.text:
        display(Markdown(p.text))
    elif p.executable_code:
        display(Markdown(f"```python\n{p.executable_code.code}\n```"))
    elif p.inline_data:
        display(Image(data=p.inline_data.data, width=800, format="png"))

```


```python
concise_search("5 largest companies by market capitalization")

```



I believe you might be referring to "market capitalization" instead of "population" when asking about companies. I will provide a bar chart of the top 5 largest companies by market capitalization.

Here are the top 5 largest companies by market capitalization as of May 21, 2025:

1.  **Microsoft:** $3.405 trillion
2.  **Nvidia:** $3.277 trillion
3.  **Apple:** $3.089 trillion
4.  **Amazon:** $2.166 trillion
5.  **Alphabet (Google):** $1.997 trillion

Now, I will create a bar chart with this data.




```python
import pandas as pd
import matplotlib.pyplot as plt

# Data for the top 5 companies by market capitalization
data = {
    'Company': ['Microsoft', 'Nvidia', 'Apple', 'Amazon', 'Alphabet (Google)'],
    'Market Cap (Trillions USD)': [3.405, 3.277, 3.089, 2.166, 1.997]
}

df = pd.DataFrame(data)

# Create the bar chart
plt.figure(figsize=(10, 6))
plt.bar(df['Company'], df['Market Cap (Trillions USD)'], color='skyblue')
plt.xlabel('Company')
plt.ylabel('Market Capitalization (Trillions USD)')
plt.title('Top 5 Largest Companies by Market Capitalization (as of May 21, 2025)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()
```



    
![png](03-structured-outputs-function-calling-tools_files/03-structured-outputs-function-calling-tools_19_3.png)
    



The bar chart showing the top 5 largest companies by market capitalization has been successfully created and displayed.


## Recap & Next Steps

**What You've Learned:**
- Structured outputs using Pydantic models for reliable data extraction and validation
- Function calling to integrate external APIs, databases, and custom business logic
- Native tools including Google Search, URL context analysis, and code execution
- Combining multiple tools for comprehensive workflows and complex problem-solving

**Key Takeaways:**
- Structured outputs ensure consistent data formats for downstream applications
- Function calling enables seamless integration with external systems and real-time data
- Native tools provide powerful capabilities without additional setup or infrastructure
- Tool combinations unlock sophisticated workflows and multi-step problem solving
- Proper validation and error handling are crucial for reliable tool interactions

**Next Steps:** Continue with [Part 4: Model Context Protocol (MCP)](https://github.com/philschmid/gemini-2.5-ai-engineering-workshop/blob/main/notebooks/04-model-context-protocol-mcp.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philschmid/gemini-2.5-ai-engineering-workshop/blob/main/notebooks/04-model-context-protocol-mcp.ipynb)

**More Resources:**
- [Structured Output Documentation](https://ai.google.dev/gemini-api/docs/structured-output?lang=python)
- [Function Calling Documentation](https://ai.google.dev/gemini-api/docs/function-calling?lang=python)
- [Grounding with Google Search](https://ai.google.dev/gemini-api/docs/grounding)
- [URL Context Tool](https://ai.google.dev/gemini-api/docs/url-context)
- [Code Execution Documentation](https://ai.google.dev/gemini-api/docs/code-execution)
