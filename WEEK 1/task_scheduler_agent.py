#Simple AI Agent: Daily Task Scheduler

def agent_task_scheduler(hour):
    # Perception: senses hour input #Decision/Action: rule-based output
    if 5 <= hour < 9:
        return "Suggest: Morning exercise 🏃‍♂️"
    elif 9 <= hour < 12:
        return "Suggest: Focus work session 💻"
    elif 12 <= hour < 14:
        return "Suggest: Lunch & Break 🍲"
    elif 14 <= hour < 18:
        return "Suggest: Meetings or deep work 📊"
    elif 18 <= hour < 21:
        return "Suggest: Family/leisure 🕺"
    else:
        return "Suggest: Rest or light reading 🌙"
    
if __name__ == "__main__":
    time_str = input("Enter current hour (0-23): ")
    try:
        hour = int(time_str)
        print(agent_task_scheduler(hour))
    except ValueError:
        print("Invalid input. Please enter a valid hour (0-23).")
      
