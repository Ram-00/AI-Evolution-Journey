# Simple FAQ Chatbot Agent (Rule-based version for Local use)
faq_responses = {
    "hours": "We're open from 9 AM to 6 PM, Monday through Saturday.",
    "reset password": "Click 'Forgot Password' on the login page and follow the instructions.",
    "location": "Our office is at 123 Main St, Downtown.",
    "contact": "You can email us at support@example.com or call +123456789."
}

def chatbot_agent(user_input):
    user_input = user_input.lower()
    for key in faq_responses:
        if key in user_input:
            return faq_responses[key]
    return "Sorry, I didn't catch that. Could you try rephrasing?"

if __name__ == "__main__":
    print("Hi! Ask me anything about our service:")
    while True:
        usr = input("> ")
        if usr.lower() in ["exit", "quit"]:
            print("Bye!")
            break
        print(chatbot_agent(usr))
