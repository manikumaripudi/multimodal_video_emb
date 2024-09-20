# import vertexai
# from vertexai.vision_models import Video, MultiModalEmbeddingModel, VideoSegmentConfig
# from pytube import YouTube
# import os
# from urllib.error import HTTPError

# # Set Google application credentials
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"D:\path\to\your\serverKey.json"

# # GCP project and location details
# PROJECT_ID = "fresh-span-400217"
# LOCATION = "us-central1"

# # Initialize the Vertex AI client
# vertexai.init(project=PROJECT_ID, location=LOCATION)

# # Step 1: Download the YouTube video using pytube
# video_url = "https://www.youtube.com/watch?v=WLQ6HyFbfKU"

# try:
#     yt = YouTube(video_url)
#     stream = yt.streams.filter(file_extension='mp4').first()
#     video_path = stream.download()
# except HTTPError as e:
#     print(f"Failed to download the video. HTTP Error: {e}")
#     video_path = None

# # Step 2: If video is successfully downloaded, load and process it
# if video_path:
#     # Load the video from the downloaded file
#     video = Video.load_from_file(video_path)

#     # Step 3: Load the multimodal embedding model
#     model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")

#     # Step 4: Generate embeddings for the video
#     embeddings = model.get_embeddings(
#         video=video,
#         video_segment_config=VideoSegmentConfig(end_offset_sec=None),  # Process the entire video
#     )

#     # Step 5: Print the video embedding details
#     print("Video Embedding:")
#     for video_embedding in embeddings.video_embeddings:
#         print(f"Embedding for segment {video_embedding.start_offset_sec} - {video_embedding.end_offset_sec}:")
#         print(video_embedding.embedding)

#     # Clean up the downloaded video file
#     os.remove(video_path)
# else:
#     print("Skipping video embedding as video download failed.")









# import yt_dlp
# import vertexai
# from vertexai.vision_models import Video, MultiModalEmbeddingModel, VideoSegmentConfig
# import os 
# from dotenv import load_dotenv
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"serverKey.json"

# load_dotenv()  
# # Initialize the Vertex AI client
# PROJECT_ID = "fresh-span-400217"
# LOCATION = "us-central1"
# vertexai.init(project=PROJECT_ID, location=LOCATION)

# # Step 1: Download the YouTube video using yt-dlp
# video_url = "https://www.youtube.com/watch?v=WLQ6HyFbfKU"  # Replace with your video URL
# ydl_opts = {
#     'format': 'best',
#     'outtmpl': 'downloaded_video.mp4'  # Save the video as 'downloaded_video.mp4'
# }

# with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#      ydl.download([video_url])

# video_path = "downloaded_video.mp4"

# # Step 2: Load the video from the downloaded file
# video = Video.load_from_file(video_path)

# # Step 3: Load the multimodal embedding model
# model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")

# # Step 4: Generate embeddings for the video
# embeddings = model.get_embeddings(
#     video=video,
#     video_segment_config=VideoSegmentConfig(end_offset_sec=None),  # Process the entire video
# )

# # Step 5: Print the video embedding details
# print("Video Embedding:")
# for video_embedding in embeddings.video_embeddings:
#     print(f"Embedding for segment {video_embedding.start_offset_sec} - {video_embedding.end_offset_sec}:")
#     print(video_embedding.embedding)

# # Clean up the downloaded video file
# os.remove(video_path)




# import vertexai
# import os 
# from vertexai.vision_models import  MultiModalEmbeddingModel, Video
# from vertexai.vision_models import VideoSegmentConfig


# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"serverKey.json"

# PROJECT_ID = "fresh-span-400217"
# location = "us-central1"
# # TODO(developer): Update project_id and location
# vertexai.init(project=PROJECT_ID, location="us-central1")

# model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")


# video = Video.load_from_file(
#     "gs://cloud-samples-data/vertex-ai-vision/highway_vehicles.mp4"
# )

# embeddings = model.get_embeddings(

#     video=video,
#     video_segment_config=VideoSegmentConfig(end_offset_sec=1),
#     contextual_text="Cars on Highway",
# )


# # Video Embeddings are segmented based on the video_segment_config.
# print("Video Embeddings:")
# for video_embedding in embeddings.video_embeddings:
#     print(
#         f"Video Segment: {video_embedding.start_offset_sec} - {video_embedding.end_offset_sec}"
#     )
#     print(f"Embedding: {video_embedding.embedding}")








import vertexai
import os
from vertexai.vision_models import MultiModalEmbeddingModel, Video

# Set up the environment
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"serverKey.json"

PROJECT_ID = "fresh-span-400217"
location = "us-central1"

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=location)

# Load the multimodal embedding model
model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")

# Load the video file (assuming the video is from a GCS bucket)
video = Video.load_from_file(
    "gs://cloud-samples-data/vertex-ai-vision/highway_vehicles.mp4"
)

# Get the embedding for the entire video as a single unit
# By setting 'end_offset_sec=None', we process the whole video
embeddings = model.get_embeddings(
    video=video,
    video_segment_config=None,  # No segmentation, single embedding for the entire video
    contextual_text="Cars on Highway",  # Optional: context for the video
)

# Output the single embedding for the entire video
print("Single Video Embedding:")
print(embeddings.video_embeddings[0].embedding)

