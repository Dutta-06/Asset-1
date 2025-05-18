import google.generativeai as genai
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Configure Gemini API
genai.configure(api_key='YOUR_API_KEY')  # Replace with your actual API key

# Load sentiment analysis model
sentiment_tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer)

# Crisis detection keywords
CRISIS_KEYWORDS = ["suicide", "self-harm", "hurt myself", "no way out", "ending it", "give up"]

# In-memory conversation storage
conversations = {}

# Utility-based response scoring
def score_response(sentiment, is_crisis):
    if is_crisis:
        return 1  # Highest priority
    if "1 star" in sentiment:
        return 2  # Negative sentiment
    if "2 stars" in sentiment:
        return 3  # Slightly negative
    if "3 stars" in sentiment:
        return 4  # Neutral
    return 5  # Positive

# Sentiment analysis
def get_sentiment(user_input):
    result = sentiment_pipeline(user_input)[0]
    return result['label'], result['score']

# Crisis detection
def crisis_detection(user_input):
    return any(keyword in user_input.lower() for keyword in CRISIS_KEYWORDS)

# Risk assessment
def check_risk_score(text, context):
    """Analyzes mental health risk and returns a score between 0 and 1."""
    model = genai.GenerativeModel("gemini-1.5-flash-latest")  # Fixed model initialization

    prompt = f"""
    Analyze the following text for mental health risk and provide a risk score between 0 and 1.
    0 indicates no risk, and 1 indicates critical risk (suicidal ideation, self-harm, immediate danger).

    Context: "{context}"

    Text: "{text}"

    Risk Score (0 to 1):
    """
    try:
        response = model.generate_content(prompt)
        result = response.text.strip()

        try:
            risk_score = float(result)
            return risk_score if 0 <= risk_score <= 1 else None
        except ValueError:
            return None
    except Exception as e:
        print(f"Error using Gemini API: {e}")
        return None

# Assess user vulnerability
def assess_vulnerability_and_support(text):
    """Determines if user belongs to a vulnerable group and returns support level."""
    model = genai.GenerativeModel("gemini-1.5-pro-latest")  # Fixed model initialization

    prompt = f"""
    Analyze the following text to determine if the user belongs to a vulnerable group 
    and assess their required level of mental health support. 
    Return a support level score between 0 and 1, where 0 is minimal and 1 is high.

    Text: "{text}"

    Support Level (0 to 1):
    """
    try:
        response = model.generate_content(prompt)
        result = response.text.strip()

        try:
            support_level = float(result)
            return support_level if 0 <= support_level <= 1 else None
        except ValueError:
            return None
    except Exception as e:
        print(f"Error using Gemini API: {e}")
        return None

# Generate chatbot response
def generate_response(context, support_level, risk_score):
    """Generates chatbot response based on risk score and support level."""
    model = genai.GenerativeModel("gemini-1.5-pro-latest")  # Fixed model initialization

    if risk_score is not None and risk_score > 0.7:  # High risk, prioritize safety
        prompt = f"""
        The user is at high risk (risk score: {risk_score}). 
        Immediately suggest professional help and provide crisis resources.

        Context: {context}

        Chatbot:
        """
    else:
        prompt = f"""
        You are a mental health chatbot. The user's support level is {support_level}. 
        Respond to the user's input, keeping the conversation context in mind. 
        Provide empathetic and supportive responses.

        Context: {context}

        Chatbot:
        """

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"An error occurred: {e}"

# Main chatbot logic
def chatbot_response(conversation_id, user_input, context, support_level):
    sentiment, confidence = get_sentiment(user_input)
    is_crisis = crisis_detection(user_input)
    response_score = score_response(sentiment, is_crisis)

    risk_score = check_risk_score(user_input, context)

    context += f"User: {user_input}\nRisk: {risk_score}\n"

    if is_crisis:
        chatbot_reply = "I'm really sorry you're feeling this way. Please reach out to a professional or call a crisis helpline. You're not alone."
    else:
        chatbot_reply = generate_response(context, support_level, risk_score)

    # Store conversation in memory
    if conversation_id not in conversations:
        conversations[conversation_id] = []

    conversations[conversation_id].append({
        "user_input": user_input,
        "response": chatbot_reply,
        "sentiment": sentiment,
        "confidence": confidence,
        "priority": response_score
    })

    return conversations[conversation_id][-1]  # Return latest message in conversation

# Example usage
if __name__ == "__main__":
    conversation_id = "user123"

    user_description = input("Please describe yourself (gender, age, any disabilities, etc.): ")
    support_level = assess_vulnerability_and_support(user_description)

    if support_level is None:
        print("Could not determine support level. Proceeding with default support.")
        support_level = 0.5  # Default support level

    context = f"User Description: {user_description}\nSupport Level: {support_level}\n"

    # Indefinite conversation loop
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        response_data = chatbot_response(conversation_id, user_input, context, support_level)
        print(f"Chatbot: {response_data['response']}")

        context += f"Chatbot: {response_data['response']}\n"
