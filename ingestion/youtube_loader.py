# This function is used to load YouTube data 
from youtube_transcript_api import YouTubeTranscriptApi

def load_youtube_data(video_id, language) -> str:
    api = YouTubeTranscriptApi()
    
    try:
        transcript_data = api.fetch(video_id, languages=[language])
        transcript_text = " ".join([entry.text for entry in transcript_data])
        # print(transcript_data)
        return transcript_text
    
    except Exception as e:
        print(f"Error fetching transcript for video {video_id}: {e}")
        return "Error fetching transcript"
