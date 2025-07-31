# AGI Decision Tree Simulation: Context-Adaptive Task Agent

def agi_agent(context):
    """Simulates different types of reasoning by an AGI-like agent."""
    if context["situation"] == "medical":
        if context["symptom"] == "fever":
            return "Recommend: Take temperature and book telehealth consult."
        else:
            return "Monitor syptoms and advise rest."
    elif context["situation"] == "finance":
        if context["event"] == "market dip":
            return "Advise: Diversify portfolio and hold off panic selling."
        else:
            return "Continue regular investment plan."
    elif context["situation"] == "daily_life":
        if context["weather"] == "rainy":
            return "Reminder: Pack an umbrella."
        else:
            return "Set up calendar for outdoor activities."
    else:
        return "General advice: Stay curious and keep learning!"

# Demo usage:
sample_context = {
    "situation": "finance",
    "event": "market dip"
}
print("AGI Agent Output:", agi_agent(sample_context))
