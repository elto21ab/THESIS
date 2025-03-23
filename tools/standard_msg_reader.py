import json, re
import tools.brikasutils as bu
from dataclasses import dataclass
from functools import partial
from datetime import datetime
from typing import List

@dataclass
class Message:
    """ An abtract message"""
    content: str
    sender: str
    timestamp: str
    seq: int
    filename: str

    def __init__(self,content = None, sender = None, timestamp = None, seq = None, filename=None, photos=None, reactions=None):
        # Mandatory
        self.content = content
        self.sender = sender

        # Optional
        self.timestamp = timestamp
        self.seq = seq
        self.filename = filename

    def __eq__(self, other):
        return self.sender == other.sender and self.content == other.content

    def __str__(self) -> str:
        return f"{self.sender}: {self.content}"

    def get_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp/1000.0)

    def as_dict(self) -> dict:
        r = self.__dict__
        r["date"] = self.get_datetime().strftime('%Y-%m-%d %H:%M:%S.%f')[:-1]
        return r

def _try_read(msg, key, msg_on_error = None):
    try:
        return msg[key]
    except:
        if msg_on_error is not None: print(msg_on_error)
        return None
    
def _parse_standard_messages_from_JSONs(filenames, limit = None, flag = None) -> list[Message]:
    """
    Returns list of Messages
    """

    if flag == "fix_facebook_encoding":
        # Fixing facebook encoding
        # Code borrowed from https://stackoverflow.com/questions/50008296/facebook-json-badly-encoded
        fix_mojibake_escapes = partial(
            re.compile(rb'\\u00([\da-f]{2})').sub,
            lambda m: bytes.fromhex(m.group(1).decode()))
        #######

    sequence = 1
    parsed_messages = []

    skip_count = 0
    fail_count = 0
    failed_files = 0
    smallest_timestamp = None
    biggest_timestamp = None
    for filename in filenames:
        if limit is not None and sequence >= limit: break
        # print("Reading", filename, "...")
        if flag == "fix_facebook_encoding":
            # Code borrowed from https://stackoverflow.com/questions/50008296/facebook-json-badly-encoded 
            with open(filename, 'rb') as binary_data:
                repaired = fix_mojibake_escapes(binary_data.read())
                data = json.loads(repaired.decode('utf8'))
            ####
        else:
            with open(filename, 'r', encoding="utf8") as jsonfile:
                data = json.load(jsonfile)

        try:
            imported_msgs = data['messages']
        except KeyError:
            print("Cannot interpret file", filename)
            failed_files += 1
            continue

        for msg in imported_msgs:
            if limit is not None and sequence >= limit: break
            try:
                cur_msg = Message(filename=filename, seq=sequence)

                # Mandatory items
                cur_msg.sender = _try_read(msg, "sender_name", "Could not read sender name for a message.")
                if cur_msg.sender is None:
                    fail_count += 1
                    continue
                
                # Optional
                cur_msg.content = _try_read(msg,'content')
                cur_msg.timestamp = _try_read(msg, "timestamp_ms")
                
                if cur_msg.timestamp is not None:
                    if smallest_timestamp is None or cur_msg.timestamp < smallest_timestamp.timestamp:
                        smallest_timestamp = cur_msg
                    if biggest_timestamp is None or cur_msg.timestamp > biggest_timestamp.timestamp:
                        biggest_timestamp = cur_msg

                parsed_messages.append(cur_msg)
                sequence += 1

            except Exception as e:
                fail_count += 1
                print("Msg failed to parse:",e)

    print(f"Read {sequence} messages from {len(filenames)-failed_files} files. Failed to read {fail_count} messages.")
    print(f"Messages ranged from {smallest_timestamp.get_datetime().strftime('%Y-%m-%d')}",
          f"to {biggest_timestamp.get_datetime().strftime('%Y-%m-%d')}")
    return parsed_messages


def get_facebook_messages_from_JSONs(filenames, limit = None) -> list[Message]:
    """
    Returns list of Messages.
    For facebook, it uses a fix for the encoding.
    """
    return _parse_standard_messages_from_JSONs(filenames, limit, flag="fix_facebook_encoding")
    
def get_whatsapp_messages_from_JSONs(filenames, limit = None) -> list[Message]:
    """
    Returns list of Messages.
    """
    return _parse_standard_messages_from_JSONs(filenames, limit)
    