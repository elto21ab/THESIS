# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "beeper-desktop-api>=4.1.296",
# ]
# ///

import os
from beeper_desktop_api import BeeperDesktop

beeper = BeeperDesktop(
    access_token=os.environ.get("BEEPER_ACCESS_TOKEN"),
    base_url="http://localhost:23373",
)

def main() -> None:
    output_path = os.path.join(os.path.dirname(__file__), "beeper_output_list.md")

    def log(out_file, text: str = "") -> None:
        print(text)
        out_file.write(text + "\n")

    with open(output_path, "w", encoding="utf-8") as out:
        for chat in beeper.chats.list():
            if chat.type != "group":
                log(out, f"\n👥 {chat.title} ({chat.network})")
                for msg in beeper.messages.list(chat_id=chat.id):
                    # print(msg.reactions) # CURRENTLY IT DOESN'T GET REACTIONS #TODO Fix it.
                    sender = "SUBJECT" if msg.is_sender else "FRIEND"  # msg.sender_name # anonymize w/ FRIEND
                    content = msg.text or f"[{msg.type}]"
                    time = msg.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    # time = msg.timestamp.strftime("%yW%W %H:%M") # This is a more compact time-format.
                    log(out, f"  [{time}] {sender}: {content}")
            else:
                log(out, f"\n👥 {chat.title} ({chat.network})")
                participants = ', '.join([p.full_name for p in chat.participants.items if p.full_name])
                log(out, f"  ({chat.participants.total}) Participants: {participants}")
                # print(chat.participants.items)
        
        

if __name__ == "__main__":
    main()
