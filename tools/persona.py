from typing import List
import standard_msg_reader as msg_reader
import re
import tiktoken
import shared_utils as utils


class PersonaEncoder:
    chats: dict
    selectedChats: dict

    def __init__(self):
        self.chats = {}
        self.selectedChats = {}
        self.nonChatModules = {}
        self.selectedNonChatModules = {}

    def parse_fb_messages(self, filenames, name_id, limit = None) -> None:
        """
        Parses Facebook Messages
        """
        msgs = msg_reader.get_facebook_messages_from_JSONs(filenames=filenames, limit=limit)
        self.chats[name_id] = list(reversed(msgs))
        print(f"Messages saved to self.chats['{name_id}']")

    def parse_wa_messages(self, filenames, name_id, limit = None) -> None:
        """
        Parses WhatsApp Messages
        """
        msgs = msg_reader.get_whatsapp_messages_from_JSONs(filenames=filenames, limit=limit)
        self.chats[name_id] = list(reversed(msgs))
        print(f"Messages saved to self.chats['{name_id}']")


    def filter_chats_empty(self):
        for nameid, chat in self.chats.items():
            filteredChat = []
            for msg in chat:
                if msg.content == "" or msg.content == None:
                    continue
                filteredChat.append(msg)
            self.chats[nameid] = filteredChat
            

    def filter_chats_regex(self, blacklist_re_patterns):
        # Instantiate filter log
        logs = {}
        for filter in blacklist_re_patterns:
            logs[filter["id"]] = 0

        for nameid, chat in self.chats.items():
            filteredChat = []
            for msg in chat:
                _excludeCurrent = False
                for filter in blacklist_re_patterns:
                    if bool(re.search(filter["pattern"] , msg.content)):
                        logs[filter["id"]] = logs[filter["id"]] + 1
                        _excludeCurrent = True
                        break
                if _excludeCurrent: continue
                filteredChat.append(msg)
            self.chats[nameid] = filteredChat

        print("Filtering")
        for key,value in logs.items():
            print(f"{key}: {value}")

    def _strinfigy_chat(chat: List[msg_reader.Message]):
        blocks = []
        for msg in chat:
            block = f"{msg.sender}: {msg.content}"
            blocks.append(block)
        return "\n".join(blocks) 

    def select_chat_limited_by_tokens(self, nameid, token_count, start_msg = 0, speed = 1):
        chat = self.chats[nameid]
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

        finalChat = []
        for i, msg in enumerate(chat):
            if i % speed == 0:
                fullText = PersonaEncoder._strinfigy_chat(finalChat)
                num_tokens = len(encoding.encode(fullText))
                if num_tokens > token_count:
                    finalChat = finalChat[:-speed]
                    break
            finalChat.append(msg)
        
        finalTokens = final_tokens = len(encoding.encode(PersonaEncoder._strinfigy_chat(finalChat)))
        self.selectedChats[nameid] = finalChat
        print(f"Selected chat {nameid} for {final_tokens} ({len(finalChat)} messages)")

    def select_chat_full(self, nameid):
        finalChat = self.chats[nameid]
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        finalTokens = final_tokens = len(encoding.encode(PersonaEncoder._strinfigy_chat(finalChat)))
        self.selectedChats[nameid] = finalChat
        print(f"Selected chat {nameid} for {final_tokens} ({len(finalChat)} messages)")

    def select_nonChat_module_full(self, nameid):
        finalModule = self.nonChatModules[nameid]
        self.selectedNonChatModules[nameid] = finalModule
        print(f"Selected module {nameid}")

    def count_chat_tokens(self, nameid):
        finalChat = self.chats[nameid]
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        finalTokens = final_tokens = len(encoding.encode(PersonaEncoder._strinfigy_chat(finalChat)))
        print(f"Chat {nameid} has {final_tokens} ({len(finalChat)} messages)")


    def count_all_selected_chat_tokens(self):
        chat_tokens = {}
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        for nameid, chat in self.selectedChats.items():
            tokens = len(encoding.encode(PersonaEncoder._strinfigy_chat(chat)))
            chat_tokens[nameid] = tokens
        return chat_tokens

    def output(self) -> str:
        finalText = ""
        for nameid, chat in self.selectedChats.items():
            finalText = finalText + PersonaEncoder._strinfigy_chat(chat)
        
        return finalText
    
    def parse_rosebud_entries(self, filename, name_id):
        """
        Parses Rosebud Diaries
        """
        with open(filename, 'r', encoding="utf-8") as file:
            md_text = file.read()

        blocks = []
        h2_nested = re.findall(r'(?<!#)##(?!#)\s[\s\S]*?---', md_text)
        for h2_n in h2_nested:
            block = {}
            title = re.findall(r'(?<!#)##(?!#)\s([\s\S]*?)[\n\r]', h2_n)
            date = re.findall(r'(?<!#)###(?!#)\s([\s\S]*?)[\n\r]', h2_n)
            content = re.findall(r'(?<!#)###(?!#)\s[\s\S]*?[\n\r]([\s\S]*?)---', h2_n)
            msgs = re.findall(r'[\s\S]*?(?:\n\n|---)', content[0])
            block['title'] = title[0]
            block['date'] = date[0]
            block['msgs'] = msgs
            blocks.append(block)

        self.nonChatModules[name_id] = blocks


        # Display the results
        total_msgs, min_msgs, max_msgs = 0, len(blocks[0]['msgs']), len(blocks[0]['msgs'])
        total_tokens, min_tokens, max_tokens = 0, utils.count_tokens(blocks[0]['msgs'][0]), utils.count_tokens(blocks[0]['msgs'][0])

        for block in blocks:
            block_msgs_count = len(block['msgs'])
            total_msgs += block_msgs_count  
            min_msgs = min(min_msgs, block_msgs_count)
            max_msgs = max(max_msgs, block_msgs_count)
            
            for msg in block['msgs']:
                tokens_count = utils.count_tokens(msg)
                total_tokens += tokens_count
                min_tokens = min(min_tokens, tokens_count)
                max_tokens = max(max_tokens, tokens_count)

        # Calculate averages
        average_msgs = total_msgs / len(blocks)
        average_tokens = total_tokens / len(blocks)

        # Print results
        print(f"Read {len(blocks)} rosebud entries.")
        print(f"Average messages per block: {round(average_msgs,2)} ({min_msgs} - {max_msgs})")
        print(f"Average tokens per block: {round(average_tokens,2)} ({min_tokens} - {max_tokens})")
