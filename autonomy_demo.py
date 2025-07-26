import datetime

def ai_agent_perception_action(input_time):
    # Perception: Sense the input (e.g., time of day)
    try:
        hour = datetime.datetime.strptime(input_time, "%H:%M").hour
    except ValueError:
        return "Invalid time format. Please use HH:MM."
    
    # Action: Decide and respond based on simple rules (rule-based autonomy)
    if 5 <= hour < 12:
        action = "Brew a strong coffee – it's morning hustle time!"
    elif 12 <= hour < 18:
        action = "Opt for a lighter blend – afternoon energy boost."
    else:
        action = "Decaf recommended – wind down for the evening."
    
    # Ethical note: In real AI, ensure diverse data to avoid biases (e.g., cultural preferences in routines)
    return f"Agent action: {action}"

# Demo usage
user_input = input("Enter time (HH:MM): ")  # Simulate sensing user input
response = ai_agent_perception_action(user_input)
print(response)

# For GitHub: This is a basic loop; expand with ML for advanced autonomy.
