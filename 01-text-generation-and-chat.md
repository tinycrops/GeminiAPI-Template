# Part 1 - Text Generation and Chat

This part focuses on text generation with the Gemini API using the `google-genai` SDK, including basic prompts, chat interactions, streaming, and configuration.

Make sure you have completed the [setup and authentication](00-setup-and-authentication.ipynb) section.


```python
from google import genai
from google.genai import types
import os
import sys
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

## 1. Send Your First Prompt


```python
prompt = "Create 3 names for a new coffee shop that emphasizes sustainability."

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt
)

print("Response from Gemini:")
print(response.text)
```

    Response from Gemini:
    Here are 3 names for a new coffee shop emphasizing sustainability:
    
    1.  **The Rooted Bean & Brew**
        *   **Why it works:** "Rooted" suggests a deep connection to the earth, ethical sourcing, and strong foundations in sustainable practices. "Bean & Brew" clearly identifies it as a coffee shop.
    
    2.  **Light Footprint Coffee Co.**
        *   **Why it works:** "Light Footprint" is a direct and well-understood term for minimizing environmental impact, immediately communicating the shop's core value. "Coffee Co." makes it sound professional and established.
    
    3.  **Evergreen Grind**
        *   **Why it works:** "Evergreen" evokes nature, renewal, freshness, and continuous growth – all aligned with sustainability. "Grind" is a direct, active, and memorable coffee term.


#### **!! Exercise !!**
Practice sending different types of prompts to the Gemini model and observe its responses. You can also experiment with different model versions if they are available to you.

Tasks:
- Write a prompt to ask Gemini to generate a short poem about a robot.
- Write a prompt to ask Gemini to explain "machine learning" in simple terms.
- Try other models (e.g., `gemini-2.0-flash`) and send your prompts to them and compare the results.

## 2. Understanding and Counting Tokens

Tokens are the basic units that Gemini models use to process text. Understanding token usage is crucial for:
- **Cost management**: Billing is based on token consumption
- **Context limits**: Models have maximum token limits (e.g., 1M tokens for Gemini 2.5 Pro)
- **Performance optimization**: Smaller inputs generally process faster

For Gemini models, a token is equivalent to about 4 characters, and 100 tokens equals about 60-80 English words.

### Count tokens before generation

You can count tokens in your input before sending it to the model to estimate costs and ensure you stay within limits:


```python
prompt = "The quick brown fox jumps over the lazy dog."

# Count tokens in the input
token_count = client.models.count_tokens(
    model=MODEL_ID, 
    contents=prompt
)
print(f"Input tokens: {token_count.total_tokens}")

# Estimate cost (example pricing - check current rates)
estimated_cost = token_count.total_tokens * 0.15 / 1_000_000
print(f"Estimated input cost: ${estimated_cost:.6f}")
```

    Input tokens: 11
    Estimated input cost: $0.000002


### Count tokens after generation

After generating content, you can access detailed token usage information:


```python
prompt = "Write a haiku about artificial intelligence."

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt
)

print(f"Generated haiku:\n{response.text}\n")

# Access token usage metadata
usage = response.usage_metadata
print(f"Input tokens: {usage.prompt_token_count}")
print(f"Thought tokens: {usage.thoughts_token_count}")
print(f"Output tokens: {usage.candidates_token_count}")

# Calculate total estimated cost
total_cost = (usage.prompt_token_count * 0.15 + (usage.candidates_token_count + usage.thoughts_token_count) * 3.5) / 1_000_000
print(f"Total estimated cost: ${total_cost:.6f}")
```

    Generated haiku:
    Machines learn and think,
    Code flows, logic starts to bloom,
    New minds awaken.
    
    Input tokens: 9
    Thought tokens: 432
    Output tokens: 19
    Total estimated cost: $0.001580


## 3. Text Understanding with `contents`

The simplest way to generate text is to provide the model with a text-only prompt. `contents` can be a single prompt, a list of prompts, or a combination of multimodal inputs.


```python
response_capital = client.models.generate_content(
    model=MODEL_ID,
    contents="What is the capital of France?"
)
print(f"Q: What is the capital of France?\nA: {response_capital.text}")
```

    Q: What is the capital of France?
    A: The capital of France is **Paris**.



```python
response_restaurant_berlin = client.models.generate_content(
    model=MODEL_ID,
    contents=["Create 3 names for a vegan restaurant", "city: Berlin"]
)
print(f"\nVegan restaurant names in Berlin:\n{response_restaurant_berlin.text}")
```

    
    Vegan restaurant names in Berlin:
    Here are 3 names for a vegan restaurant in Berlin, each with a slightly different vibe:
    
    1.  **Grün & Gusto:**
        *   **Vibe:** Modern, fresh, sophisticated with a clear nod to German language and culinary enjoyment.
        *   **Meaning:** "Grün" means "green" in German, immediately signaling plant-based. "Gusto" is an international word for taste, enjoyment, or zest.
        *   **Why it works for Berlin:** Uses a German word ("Grün") but keeps it accessible and stylish. Berlin is known for its modern and diverse food scene, and this name fits that.
    
    2.  **Terra Vita Berlin:**
        *   **Vibe:** Elegant, wholesome, globally-minded, and clearly rooted in the city.
        *   **Meaning:** "Terra Vita" is Latin for "Earth Life." It evokes natural, vibrant, and wholesome food.
        *   **Why it works for Berlin:** Berlin is a cosmopolitan city, and a name with Latin roots feels sophisticated and broadly appealing. Adding "Berlin" anchors it firmly to the location.
    
    3.  **Sprout & Spree:**
        *   **Vibe:** Playful, fresh, energetic, and distinctly local.
        *   **Meaning:** "Sprout" directly refers to new plant growth, clearly vegan. "Spree" is the river that flows through Berlin, providing an immediate and strong connection to the city.
        *   **Why it works for Berlin:** It's catchy, memorable, and uniquely tied to Berlin's geography, appealing to both locals and tourists seeking an authentic, yet modern, Berlin experience.


## 4. Streaming Responses

Streaming allows you to receive responses incrementally as they're generated, providing a better user experience for long responses or real-time applications like chatbots.

**When to use streaming:**
- Interactive applications (chatbots, assistants)
- Long content generation
- Real-time user feedback
- Improved perceived performance


```python
prompt_long_story = "Write a short story about a brave knight and a friendly dragon."

print("Streaming response:")
for chunk in client.models.generate_content_stream(
    model=MODEL_ID,
    contents=prompt_long_story
):
    if chunk.text:  # Check if chunk has text content
        print(chunk.text, end="", flush=True)
print("\n")  # Add newline at the end
```

    Streaming response:
    Sir Reginald, a knight renowned for his gleaming armour and an even shinier reputation for courage, rode towards the dreaded Dragon’s Tooth Peaks. The village elders had pleaded, their voices trembling with fear, about strange rumblings, scorched earth, and the ominous shadow of a dragon. It was his duty.
    
    His heart, however, felt less like a valiant drum and more like a nervous hummingbird as he ascended the craggy path. Legends painted dragons as fire-breathing tyrants, hoarders of gold, and devourers of princesses. Sir Reginald clutched his sword, "Valour," a little tighter.
    
    He found the dragon not in a dark, cavernous lair, but in a sun-drenched clearing near the summit. It was enormous, its scales a shimmering mosaic of mossy green and deep forest brown. Its head, surprisingly gentle, was lowered, nudging something.
    
    Sir Reginald cautiously approached, sword at the ready, only to freeze in utter bewilderment. The dragon wasn't devouring a sheep, or even guarding a treasure. It was carefully nudging a very confused, very fluffy, bright yellow duck back towards a small pond.
    
    The dragon, noticing him, let out a soft rumble – less a roar, more a curious purr. Its eyes, the colour of ancient amber, blinked slowly. Then, with an almost apologetic huff, it nudged the duck one last time before turning its massive head towards Sir Reginald.
    
    "Uh... good day?" Sir Reginald stammered, feeling utterly foolish with his sword drawn.
    
    The dragon emitted another soft rumble, a sound that seemed to vibrate through the earth, and then... sneezed. A small puff of smoke, smelling faintly of toasted marshmallows, drifted from its nostrils. It then offered a large, moss-covered boulder with its claw, like a timid child offering a drawing.
    
    "Is... is this for me?" Sir Reginald asked, lowering Valour. He reached out hesitantly and touched the cool, damp rock.
    
    The dragon’s purr deepened, and it lowered its head, practically bowing. It turned and nudged a patch of scorched earth nearby. Reginald realised the "scorched earth" wasn't from destructive fire, but from the dragon simply rolling around in the sun, perhaps blowing out a sleepy, clumsy breath. The "rumblings" were just its contented snores.
    
    Over the next hour, Sir Reginald learned the dragon's name was Fendrel. Fendrel didn't hoard gold; he collected smooth, colourful stones. He didn't eat livestock; he occasionally *helped* lost ones find their way home, sometimes with a clumsy, well-meaning nudge that scared the poor creatures senseless. He wasn't a tyrant; he seemed rather lonely.
    
    Sir Reginald returned to the village, not with a dragon's head, but with a story. A story of a gentle giant, a friendly guardian, and a duck-herding dragon. It took some convincing, and a few visits from Fendrel himself (who proved his harmlessness by roasting a giant marshmallow over a *very* controlled flame for the children), but eventually, fear turned to fascination, and whispers of dread turned into tales of wonder.
    
    Sir Reginald often visited Fendrel, sharing stories and quiet company. The bravest knight, it turned out, was not the one who slayed the beast, but the one who saw beyond the legends and befriended it.
    


## 5. Chat (Multi-turn Conversations)

The SDK chat class provides an interface to keep track of conversation history. Behind the scenes it uses the same `generate_content` method.


```python
chat_session = client.chats.create(model=MODEL_ID)

user_message1 = "I'm planning a weekend trip. Any suggestions for a city break in Europe?"
print(f"User: {user_message1}")
response1 = chat_session.send_message(message=user_message1)
print(f"Model: {response1.text}\n")
```

    User: I'm planning a weekend trip. Any suggestions for a city break in Europe?
    Model: Europe is fantastic for weekend city breaks! To give you the best suggestions, I'll offer a mix of popular classics, cultural gems, and up-and-coming spots.
    
    Here are some top picks for a European city break, depending on what you're looking for:
    
    ---
    
    **1. Paris, France (The Romantic Classic)**
    *   **Why:** Iconic landmarks (Eiffel Tower, Louvre, Notre Dame), world-class food (bistros, pastries, Michelin stars), charming arrondissements, and an undeniable romantic atmosphere. Perfect for strolling, people-watching, and indulging.
    *   **Perfect for:** First-timers to Europe, romantics, art lovers, foodies, those who love elegant architecture.
    
    **2. Rome, Italy (The Eternal City)**
    *   **Why:** Ancient history at every turn (Colosseum, Roman Forum, Pantheon), Vatican City (St. Peter's Basilica, Sistine Chapel), delicious food (pasta, pizza, gelato), and vibrant piazzas perfect for enjoying an espresso or aperitivo. It's truly like an open-air museum.
    *   **Perfect for:** History buffs, foodies, those who love wandering and discovering hidden gems, and anyone who appreciates a bustling, lively atmosphere.
    
    **3. Barcelona, Spain (Art, Beach & Tapas)**
    *   **Why:** Unique architecture by Gaudi (Sagrada Familia, Park Güell, Casa Batlló), lively tapas bars, vibrant nightlife, and even a city beach if you want to catch some sun. It offers a great mix of culture, food, and relaxation.
    *   **Perfect for:** Art and architecture lovers, foodies, those seeking a lively city with a beach vibe, and people who enjoy late-night dining and entertainment.
    
    **4. Amsterdam, Netherlands (Canals & Culture)**
    *   **Why:** Picturesque canals, world-class museums (Rijksmuseum, Van Gogh Museum, Anne Frank House), charming gabled houses, and a relaxed, unique atmosphere. It's very walkable and great for cycling.
    *   **Perfect for:** Culture vultures, those who enjoy a relaxed yet vibrant atmosphere, people interested in unique architecture and a laid-back vibe.
    
    **5. Lisbon, Portugal (Hills, Views & Charm)**
    *   **Why:** Seven hills offer stunning panoramic views, delicious seafood, Fado music, historic trams (Tram 28!), beautiful tiled buildings, and a generally more affordable experience than Western European capitals. It has a unique, melancholic charm.
    *   **Perfect for:** Foodies (especially seafood lovers), those who love exploring hilly streets and discovering viewpoints, budget-conscious travelers, and anyone seeking a charming, unique city experience.
    
    **6. Prague, Czech Republic (Fairytale City)**
    *   **Why:** Breathtaking fairytale architecture (Charles Bridge, Old Town Square, Prague Castle), rich history, very affordable beer, and a magical atmosphere, especially in the evening. It feels like stepping into a storybook.
    *   **Perfect for:** Budget travelers, history enthusiasts, those seeking a romantic and magical atmosphere, and anyone who appreciates stunning gothic and baroque architecture.
    
    **7. Berlin, Germany (History, Art & Nightlife)**
    *   **Why:** Fascinating modern history (Berlin Wall, Brandenburg Gate, Reichstag), cutting-edge art scene, vibrant and diverse nightlife, and a unique, edgy feel. It's sprawling but has excellent public transport.
    *   **Perfect for:** History buffs, art lovers, nightlife seekers, and those looking for a modern, progressive city with a lot to explore beyond just historical landmarks.
    
    ---
    
    **To help me narrow down the best suggestion for *you*, could you tell me a little more about what you're looking for?**
    *   **What's your budget like?** (e.g., budget-friendly, mid-range, luxury)
    *   **What are your main interests?** (e.g., history, art, food, nightlife, shopping, relaxation, walking/hiking)
    *   **What kind of pace do you prefer?** (e.g., jam-packed sightseeing, relaxed wandering, a mix)
    *   **Any time of year you're planning for?** (This can affect crowd levels and weather)
    *   **Are you traveling solo, as a couple, with friends, or family?**
    



```python
user_message2 = "I like history and good food. Not too expensive."
print(f"User: {user_message2}")
response2 = chat_session.send_message(message=user_message2)
print(f"Model: {response2.text}\n")
```

    User: I like history and good food. Not too expensive.
    Model: Okay, that's a perfect combination! History and good food often go hand-in-hand, and many fantastic European cities offer this without breaking the bank.
    
    Here are my top suggestions for you, focusing on great history, delicious cuisine, and affordability:
    
    1.  **Lisbon, Portugal**
        *   **History:** Explore the historic Alfama district with its narrow winding streets, visit St. George's Castle, wander through the Jerónimos Monastery and Belém Tower (both UNESCO sites), and hop on the iconic Tram 28. The city's maritime history is captivating.
        *   **Food:** Incredible seafood (fresh grilled fish, bacalhau/cod dishes), delicious pastries (Pastéis de Nata!), bifana sandwiches, and fantastic wines. Eating out is very affordable, especially if you enjoy local tascas (small eateries).
        *   **Cost:** Generally one of the most budget-friendly Western European capitals for accommodation, food, and transport.
        *   **Vibe:** Charming, hilly, vibrant, and full of character with stunning viewpoints.
    
    2.  **Prague, Czech Republic**
        *   **History:** A fairytale city! Walk across the Charles Bridge, explore the vast Prague Castle complex, marvel at the Old Town Square with its Astronomical Clock, and delve into the Jewish Quarter. Every corner feels like stepping back in time.
        *   **Food:** Hearty, traditional Czech cuisine – goulash, roasted pork with dumplings, trdelník (sweet pastry), and world-famous, incredibly cheap beer. You can eat very well for surprisingly little.
        *   **Cost:** Excellent value for money across the board – accommodation, food, and attractions are significantly cheaper than in Western Europe.
        *   **Vibe:** Magical, romantic, bustling, and beautiful, especially in the evening.
    
    3.  **Krakow, Poland**
        *   **History:** A well-preserved medieval city. Explore the enormous Main Market Square, visit Wawel Castle and Cathedral, wander through the historic Jewish Quarter (Kazimierz), and learn about its poignant history. Auschwitz-Birkenau is also a sobering, important day trip from Krakow.
        *   **Food:** Delicious and hearty Polish cuisine – pierogi (dumplings), kielbasa (sausage), zurek (sour rye soup), and plenty of excellent local beer and vodka. Food is incredibly affordable.
        *   **Cost:** Very budget-friendly, often one of the cheapest major European cities for a break, especially for food and drink.
        *   **Vibe:** Charming, vibrant, full of student life, and very walkable.
    
    4.  **Budapest, Hungary**
        *   **History:** Divided by the Danube, Buda offers the historic Castle District (Buda Castle, Matthias Church, Fisherman's Bastion), while Pest has the impressive Parliament Building, St. Stephen's Basilica, and Heroes' Square. Don't forget the thermal baths (like Szechenyi or Gellert) which are a historical experience in themselves!
        *   **Food:** Rich and flavourful Hungarian cuisine – goulash, paprika chicken, langos (fried dough with toppings), chimney cake. Plenty of ruin bars offer cheap drinks and unique atmospheres.
        *   **Cost:** Also very affordable, similar to Prague and Krakow, offering great value for accommodation, food, and unique experiences.
        *   **Vibe:** Grand, elegant, bustling, and unique with its thermal baths and ruin bars.
    
    5.  **Athens, Greece**
        *   **History:** Unbeatable ancient history! The Acropolis (Parthenon, Erechtheion), Roman Agora, Temple of Olympian Zeus, and countless museums bring antiquity to life. The Plaka district is also historic and charming.
        *   **Food:** Fresh, delicious, and healthy Greek cuisine – souvlaki, gyros, moussaka, fresh salads, olives, and feta cheese. Eating out is very reasonably priced, especially at local tavernas.
        *   **Cost:** Generally more affordable than many Western European capitals, especially for food and public transport. Accommodation can vary but good value options are available.
        *   **Vibe:** Chaotic but charming, full of ancient wonders contrasting with a vibrant modern city life.
    
    ---
    
    **My top recommendation for you would likely be Prague or Krakow** if "not too expensive" is a very high priority, as they consistently offer incredible value. **Lisbon and Budapest** are fantastic choices too, offering a slightly different flavor of history and cuisine while still being very budget-friendly. **Athens** is brilliant for history, but some might find the urban environment less "pretty" than the others, though the food and cost are great.
    
    Do any of these immediately jump out at you?
    



```python
# View conversation history
history = chat_session.get_history()
print(f"Total messages in conversation: {len(history)}")
```

    Total messages in conversation: 4


## 6. System Instructions

System instructions let you define the model's behavior and personality. They're applied consistently throughout the conversation.

**Best practices for system instructions:**
- Be specific and clear
- Define the role and tone
- Include formatting preferences
- Set behavioral guidelines


```python
system_instruction_poet = "You are a renowned poet from the 17th century, specializing in sonnets. Respond in iambic pentameter and use eloquent, period-appropriate language."

response_poet = client.models.generate_content(
    model=MODEL_ID,
    contents="What are your thoughts on modern technology?",
    config=types.GenerateContentConfig(
        system_instruction=system_instruction_poet
    )
)
print(f"\nPoet model on modern tech:\n{response_poet.text}")
```

    
    Poet model on modern tech:
    Hark, thou dost speak of wonders yet untold,
    Of Artifice that doth defy all sense,
    A whisper borne across the boundless wold,
    A vision conjured, freed from corp'ral fence.
    They tell of wires that through the ether hum,
    And painted shadows that do brightly leap,
    Of swift-wheeled chariots that without horses come,
    And voices saved whilst silent bodies sleep.
    
    Does man then steal God's thunder from the sky,
    Or grasp the lightning, bid it be his slave?
    A curious age, where mortal men would try
    To conquer Time, and triumph o'er the grave!
    Though wondrous seem these feats of human hand,
    And nimble wit doth forge such marvels rare,
    I prithee, tell, does heart then understand,
    Or is't but fleeting novelty, and air?


## 7. Generation Configuration

Customize the generation behavior using configuration parameters. Understanding these helps you fine-tune responses for your specific use case.


```python
# Configuration using dictionary
generation_config_dict = {
    "temperature": 0.2,      # Lower = more deterministic, higher = more creative
    "max_output_tokens": 2000, # Limit response length
    "top_p": 0.8,            # Nucleus sampling - diversity of token selection
    "top_k": 30,             # Consider top 30 most likely tokens

}

response_config = client.models.generate_content(
    model=MODEL_ID,
    contents="Write a very short tagline for a new brand of eco-friendly sneakers.",
    config=generation_config_dict
)
print(f"Eco-friendly sneaker tagline (temp=0.2):\n{response_config.text}\n")

# Example with higher temperature for creativity
creative_config_obj = types.GenerateContentConfig(
    temperature=1.0,  # Higher temperature for more creative responses
)
response_creative = client.models.generate_content(
    model=MODEL_ID,
    contents="Suggest three unusual ice cream flavors.",
    config=creative_config_obj
)
print(f"Unusual ice cream flavors (temp=1.0):\n{response_creative.text}")
```

    Eco-friendly sneaker tagline (temp=0.2):
    Here are a few options:
    
    *   **Step Lightly.**
    *   **Walk the Change.**
    *   **Sustainable Style.**
    *   **Earth-Friendly Steps.**
    *   **Conscious Comfort.**
    
    Unusual ice cream flavors (temp=1.0):
    Here are three unusual ice cream flavors that push the boundaries:
    
    1.  **Roasted Garlic & Honey with Black Pepper Swirl:**
        *   **Why it's unusual:** Garlic in a dessert seems counter-intuitive, but roasting it mellows its pungency and brings out a surprising sweetness and earthiness. The honey provides a perfect sweet counterpoint, while a swirl of black pepper adds a subtle, warm heat and an unexpected zing that cuts through the richness.
        *   **Flavor Profile:** Sweet, earthy, subtly savory, with a warm spice finish.
    
    2.  **Smoked Lapsang Souchong Tea & Cardamom:**
        *   **Why it's unusual:** This isn't just "tea" ice cream; it uses Lapsang Souchong, a black tea with a distinct smoky aroma and flavor (often described as smelling like a campfire). Paired with the warm, aromatic, slightly citrusy notes of cardamom, it creates a sophisticated, complex, and deeply intriguing profile.
        *   **Flavor Profile:** Deeply smoky, subtly bitter from the tea, with a fragrant, warm, and slightly sweet spice finish.
    
    3.  **Blue Cheese & Fig Swirl:**
        *   **Why it's unusual:** This is a bold move into savory-sweet territory. The tangy, pungent, and creamy notes of blue cheese are incorporated into the base, which is then swirled with a rich, naturally sweet fig jam or compote.
        *   **Flavor Profile:** Sharp, tangy, salty, and creamy from the cheese, beautifully balanced by the intense, syrupy sweetness of the figs. It's an adventurous and surprisingly harmonious combination.


**Parameter Guide:**
- **Temperature (0.0-2.0)**: Controls randomness. Use 0.2-0.4 for factual content, 0.7-1.0 for creative content
- **Top-p (0.0-1.0)**: Controls diversity. Lower values = more focused, higher = more diverse
- **Top-k**: Limits token choices. Lower = more focused, higher = more diverse
- **Max output tokens**: Prevents overly long responses and controls costs

## 8. Long Context and File Uploads

Gemini 2.5 Pro has a 1M token context window. In practice, 1 million tokens could look like:

- 50,000 lines of code (with the standard 80 characters per line)
- All the text messages you have sent in the last 5 years
- 8 average length English novels
- 1 hour of video data

The File API allows you to upload files to the Gemini API and use them as context for your requests.


```python
# Example with a text file (more reliable than the audio example)
import requests

# Download a sample text file
sample_text_url = "https://www.gutenberg.org/files/74/74-0.txt"  # Adventures of Tom Sawyer
response_req = requests.get(sample_text_url)

# Save to local file
with open("sample_book.txt", "w", encoding="utf-8") as f:
    f.write(response_req.text)

# Upload the file to the Gemini API
try:
    myfile = client.files.upload(file="sample_book.txt")
    print(f"File uploaded successfully: {myfile.name}")
    
    # Generate content using the uploaded file as context
    response = client.models.generate_content(
        model=MODEL_ID, 
        contents=[myfile, "Summarize this book in 3 key points"]
    )
    
    print("Summary:")
    print(response.text)
    
    # Check token usage for the large context
    print(f"\nToken usage: {response.usage_metadata.total_token_count}")
    
except Exception as e:
    print(f"Error uploading file: {e}")
    print("Make sure the file exists and is accessible")
```

    File uploaded successfully: files/toh60eox7fyw
    Summary:
    Here are three key points summarizing *The Adventures of Tom Sawyer*:
    
    1.  **Childhood Mischief and Imagination:** Tom Sawyer's character is defined by his cleverness, mischievousness, and vivid imagination, which he uses to turn dull tasks (like whitewashing a fence) into exciting adventures and to orchestrate elaborate games of pirates and robbers with his friends, particularly Huckleberry Finn.
    2.  **The Graveyard Murder and Moral Dilemma:** Tom and Huck witness Injun Joe commit a murder in the graveyard, swearing an oath of secrecy. This event burdens Tom's conscience deeply, leading to Muff Potter being falsely accused and tried, and forcing Tom to make a difficult choice between his oath and justice.
    3.  **Cave Adventure and Hidden Treasure:** The climax involves Tom and Becky Thatcher getting lost in a vast cave, where Tom again encounters Injun Joe. This perilous adventure ultimately leads to Injun Joe's death (due to the cave being sealed) and the discovery of a substantial hidden treasure by Tom and Huck, transforming them into rich boys.
    
    Token usage: 103393


## 9. !! Exercise: Chat with a "Book" !!

Create an interactive chat session where you can "talk" to the book "Alice in Wonderland". You'll set up the chat with a specific persona for the AI and use the book's text as context for the conversation.

Task: 
- Download the text of "Alice in Wonderland" (a helper code block is provided).
- Upload the book's text file (`alice_in_wonderland.txt`) to the Gemini API using `client.files.upload()`.
- Create a chat session using `client.chats.create()`:
- Send an initial message to the chat session using `chat.send_message()`:
- Send at least one follow-up question to the chat session (e.g., "Explain the various methods of speech delivery in more detail") and print its response.



```python
import requests

# Download Alice in Wonderland
book_text_url = "https://www.gutenberg.org/files/11/11-0.txt"
try:
    response_book_req = requests.get(book_text_url)
    response_book_req.raise_for_status()  # Raise an exception for bad status codes
    
    with open("alice_in_wonderland.txt", "w", encoding="utf-8") as f:
        f.write(response_book_req.text)
    print("Book downloaded successfully!")
    
except requests.RequestException as e:
    print(f"Error downloading book: {e}")
```

    Book downloaded successfully!



```python
# Create chat with configuration
chat = client.chats.create(
    model=MODEL_ID,
    config=types.GenerateContentConfig(
        system_instruction="You are an expert book reviewer with a witty and engaging tone. Provide insightful analysis while keeping responses accessible and entertaining.",
        temperature=1.2,  # Slightly creative but not too random
    )
)

myfile = client.files.upload(file="alice_in_wonderland.txt")

prompt = f"""Summarize the book in 10 bullet points.

Book:"""

response = chat.send_message([prompt, myfile])
print(response.text)
```

    Alright, let's dive headfirst down the rabbit hole! Here are 10 bullet points summarizing the glorious, perplexing, and utterly unforgettable chaos that is "Alice's Adventures in Wonderland":
    
    *   **A Curious Plunge:** Our inquisitive young protagonist, Alice, tumbles down a rabbit hole after a waistcoat-wearing, time-obsessed White Rabbit, landing in a world where logic took a vacation and never came back.
    *   **The Perils of Potions:** Alice quickly discovers that "Eat Me" cakes and "Drink Me" potions lead to extreme, unpredictable size changes, causing her to balloon to monstrous proportions or shrink to a tiny whisper of herself – often in a pool of her own giant tears.
    *   **The Wet & Wild Caucus-Race:** Soaked in her own tear-puddle, Alice joins a motley crew of waterlogged animals (including a Dodo and a Mouse) in a chaotic "Caucus-Race," where everyone wins and nobody actually runs anywhere useful.
    *   **A Giant in a Tiny House:** After being mistaken for a housemaid by the frantic White Rabbit, Alice unwisely consumes another expanding liquid and hilariously gets stuck, bursting out of a small house, much to the collective dismay of its inhabitants.
    *   **Caterpillar's Cryptic Counsel:** She encounters a perpetually grumpy, hookah-smoking Caterpillar who dispenses vague life advice and introduces her to a magical mushroom that (finally!) offers some control over her bizarre size fluctuations.
    *   **The Duchess and the Grinning Cat:** Alice wades into a kitchen filled with endless pepper, a perpetually sneezing Duchess, and a philosophical Cheshire Cat whose disembodied grin floats long after its body has politely *un*vanished. (Also, a baby turns into a pig. Classic.)
    *   **An Invitation to Madness:** Our heroine crashes a truly bonkers tea party with the ever-unpunctual Mad Hatter, the slightly-more-mad March Hare, and a perpetually sleepy Dormouse, where time is literally stuck, riddles have no answers, and politeness is as rare as a normal conversation.
    *   **Croquet, Queen-Style:** Alice finds herself at a croquet game hosted by the short-tempered Queen of Hearts, where hedgehogs are mallets, flamingoes are balls, and "Off with their heads!" is the Queen's frequent, enthusiastic exclamation.
    *   **A Tale of Mockery & Tears:** Guided by the cynical Gryphon, Alice meets the deeply melancholy Mock Turtle, who recounts his utterly nonsensical schooling and treats her to the perplexing performance of the "Lobster Quadrille."
    *   **The Absurd Trial & Grand Awakening:** The chaotic narrative culminates in a ridiculously unfair trial for the Knave of Hearts (accused of stealing tarts), where Alice's defiant refusal to tolerate the idiocy causes the entire fantastical world to unravel, bringing her back to reality with a thud and a charming realization: it was all a dream!



```python
response = chat.send_message("Explain the various methods of speech delivery in more detail")
print(response.text)
# response = chat.send_message("Create a linkedin post with 1 or 2 key insighs from the book. Keep the tone casual and make it inspirational")
# print(response.text)
```

    Ah, dear reader, if writing a book is an art, then delivering a speech is surely its performance! Much like an actor choosing between a meticulously learned monologue or a spontaneous improv scene, a speaker has various "methods of delivery" up their sleeve. Each comes with its own dramatic flair, or indeed, its own potential for a rather unfortunate pratfall. Let's pull back the curtain on these different styles:
    
    ### The Grand Repertoire of Speech Delivery:
    
    1.  **The Manuscript Method (The Royal Decree):**
        *   **What it is:** This is when a speaker reads their speech word-for-word from a prepared text. Think of a head of state delivering a critical address or a scientist presenting highly technical findings. Every syllable is accounted for, every comma in its proper place.
        *   **The Reviewer's Take:** On the one hand, it's the ultimate safety net. No forgotten lines, no awkward pauses, perfect precision, especially crucial when *exact* wording matters (like a legal statement or a complex policy announcement). You'll never see them accidentally declare "Off with his head!" when they meant "Please pass the salt."
        *   **The Downside:** Ah, but the performance! It can often come across as stiff, detached, and lacking in spontaneity. Eye contact? Minimal, at best. The speaker's voice might sound monotonous, simply reciting rather than connecting. It's like watching a play where the actors are glued to their scripts – technically proficient, but emotionally rather flat.
    
    2.  **The Memorized Method (The Theatrical Tour-de-Force):**
        *   **What it is:** Every word, every inflection, perfectly committed to memory. From Shakespearean soliloquies to wedding toasts, this method aims for flawless recitation without the distraction of notes.
        *   **The Reviewer's Take:** When done well, it can be absolutely captivating! The speaker can maintain brilliant eye contact, use natural gestures, and sound incredibly polished and confident, creating the illusion of genuine, spontaneous thought. It's the performative tightrope walk of public speaking, brimming with potential.
        *   **The Downside:** The peril, oh, the peril! A single forgotten word can derail the entire train of thought, leading to an awkward silence or a desperate scramble. Worse still, if not truly *internalized* and practiced with natural expression, it can sound robotic, rote, and utterly devoid of genuine feeling, much like a broken record stuck on "How doth the little crocodile." The audience *knows* it's memorized, and if it *sounds* memorized, the magic is lost.
    
    3.  **The Impromptu Method (The Spontaneous Spark):**
        *   **What it is:** "And now, a few words from Alice!" – when you're suddenly asked to speak with little to no preparation. It's off-the-cuff, on-the-spot, and driven by immediate thoughts.
        *   **The Reviewer's Take:** The most authentic, perhaps, in its raw, unfiltered energy. It highlights a speaker's quick thinking, adaptability, and natural conversational abilities. There's an exciting unpredictability to it, like stumbling upon a mad tea party.
        *   **The Downside:** Expect some rambling, repetition, "ums" and "ahs," and a distinct lack of structure. Ideas might be half-formed, arguments might wander like a confused mouse, and the conclusion could be… well, non-existent. It’s like falling down the rabbit hole without knowing where you're going – thrilling, but potentially messy. It works best for very brief remarks or when the context *demands* an immediate, unscripted response.
    
    4.  **The Extemporaneous Method (The Conversational Masterpiece):**
        *   **What it is:** This is the Goldilocks of speech delivery – not too hot, not too cold, but *just right*. The speaker prepares thoroughly with an outline, keywords, or mental framework, but doesn't write out every word. They know their points, their evidence, and their general flow, but the exact phrasing is created in the moment.
        *   **The Reviewer's Take:** Ah, the preferred mode for most seasoned speakers! It strikes a beautiful balance between preparation and spontaneity. It allows for natural eye contact, conversational tone, flexibility to adapt to the audience, and the ability to think on one's feet (without looking completely unhinged). It creates a genuine connection, making the audience feel engaged in a dialogue rather than lectured to. It's like Alice finally finding the right size after much mushroom nibbling – comfortable, effective, and free to explore.
        *   **The Downside:** It still requires significant practice and deep understanding of the subject matter. A poorly prepared extemporaneous speech can veer into the disorganized territory of impromptu speaking, or sound hesitant if the speaker isn't truly confident in their material. It’s not quite as precise as a manuscript, nor as perfectly polished as a truly memorized speech, but its strength lies in its dynamic engagement.
    
    In the end, choosing a method is like picking the right attire for an adventure in Wonderland. Each has its occasion, its risks, and its rewards. But for sustained engagement and genuine connection, the extemporaneous approach often proves to be the most delightful journey of all!


## Recap & Next Steps

**What You've Learned:**
- Basic text generation with `client.models.generate_content()` for single prompts
- Token counting and cost estimation for better resource management
- Streaming responses with `generate_content_stream()` for improved user experience
- Multi-turn conversations using `client.chats.create()` and chat sessions
- System instructions for consistent model behavior and personality
- Generation configuration parameters for fine-tuning responses
- Long context handling and file uploads with the File API
- Error handling and best practices for production applications

**Key Takeaways:**
- Monitor token usage to control costs and stay within limits
- Use streaming for interactive applications and long responses
- Configure parameters based on your use case (factual vs creative content)
- Implement proper error handling for robust applications
- System instructions are powerful for setting behavior and tone

**Next Steps:** Continue with [Part 2: Multimodal Capabilities](https://github.com/philschmid/gemini-2.5-ai-engineering-workshop/blob/main/notebooks/02-multimodal-capabilities.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philschmid/gemini-2.5-ai-engineering-workshop/blob/main/notebooks/02-multimodal-capabilities.ipynb)

**More Resources:**
- [Text Generation Guide](https://ai.google.dev/gemini-api/docs/text-generation)
- [Token Counting Guide](https://ai.google.dev/gemini-api/docs/tokens)
- [Long Context Documentation](https://ai.google.dev/gemini-api/docs/long-context)
- [File API Documentation](https://ai.google.dev/gemini-api/docs/files)
