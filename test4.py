import asyncio, json
from google import genai
from google.genai import types

key = json.load(open('config/api_keys.json'))['gemini_api_key']
client = genai.Client(api_key=key, http_options={'api_version': 'v1beta'})

async def test():
    try:
        config = types.LiveConnectConfig(response_modalities=['AUDIO'], tools=[{'function_declarations': [{'name': 'open_app', 'description': 'opens app Spotify', 'parameters': {'type': 'OBJECT', 'properties': {'app': {'type': 'STRING'}}, 'required': ['app']}}]}])
        async with client.aio.live.connect(model='models/gemini-2.5-flash-native-audio-preview-12-2025', config=config) as s: 
            await s.send_client_content(turns={'parts': [{'text': 'Open app Spotify'}]}, turn_complete=True)
            async for resp in s.receive():
                if resp.tool_call:
                    print("TOOL CALL RECEIVED:", resp.tool_call.function_calls)
                    break
    except Exception as e:
        print('Failed:', e)

asyncio.run(test())
