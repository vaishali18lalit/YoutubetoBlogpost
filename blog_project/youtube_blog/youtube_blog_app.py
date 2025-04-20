import re
from youtube_transcript_api import YouTubeTranscriptApi
import openai
import requests
from serpapi import GoogleSearch
import os
import cv2
import yt_dlp
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import io

openai.api_key = "" #----------------------------

def process_images_and_create_mapping(keywords, save_dir, save_dir1, image_to_base64_func):
    """
    Process images from two directories (save_dir and save_dir1) based on the given keywords.
    Creates a mapping of keyword to image paths and base64-encoded images.

    :param keywords: List of keywords to search for in image file names.
    :param save_dir: Directory where images from Google are saved.
    :param save_dir1: Directory where images from YouTube are saved.
    :param image_to_base64_func: Function to convert image to base64 (e.g., `image_to_base64`).
    
    :return: A dictionary mapping each keyword to a list of image paths and base64 encodings.
    """
    mapping = {}

    # Initialize the mapping dictionary with empty lists for each keyword
    for keyword in keywords:
        mapping[keyword] = []

        # Check the images in the Google image directory (save_dir)
        for file_name in os.listdir(save_dir):
            # Check if the normalized keyword appears in the file name
            if keyword in file_name.lower():
                full_path = os.path.join(save_dir, file_name)
                encode_image_to_base64 = image_to_base64_func(full_path)
                # Add the image path and empty base64 encoding to the mapping
                mapping[keyword].append({
                    "path": full_path,
                    "base64":encode_image_to_base64   # Base64 will be added later if needed
                })

        # Check the images in the YouTube image directory (save_dir1)
        for file_name in os.listdir(save_dir1):
            if keyword in file_name.lower():
                full_path = os.path.join(save_dir1, file_name)
                # Convert the image to base64 and add to the mapping
                encode_image_to_base64 = image_to_base64_func(full_path)
                mapping[keyword].append({
                    "path": full_path,
                    "base64": encode_image_to_base64
                })

    print("Mapping:", mapping)
    return mapping


def save_google_images(keywords, save_dir):
    """
    Fetch images for each keyword and save them to the specified directory.

    :param keywords: List of keywords for which to fetch images.
    :param save_dir: Directory where the images will be saved.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    for keyword in keywords:
        # Fetch image URL based on the keyword (use your existing fetch_google_images function)
        image_url = fetch_google_images(keyword)

        if image_url:
            # Send a request to download the image
            response = requests.get(image_url)
            
            # Check if the request was successful
            if response.status_code == 200:
                # Generate the image file path
                image_path = os.path.join(save_dir, f"{keyword}.jpg")
                
                # Save the image to the specified path
                with open(image_path, "wb") as f:
                    f.write(response.content)
                
                print(f"Saved: {image_path}")
            else:
                print(f"Failed to download image for {keyword}, Status Code: {response.status_code}")
        else:
            print(f"No image URL found for keyword: {keyword}")


def generate_quiz_questions(transcript):
    prompt = f"""
    Generate 10 quiz questions from the following transcript:
    {transcript}
    Provide questions in this format:
    1. Question? 
       A) Option 1
       B) Option 2
       C) Option 3
       D) Option 4
       Answer: B
    """
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a quiz generator AI.Ask Challenging questions from the video transcript"},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


def image_to_base64(image_path):
    with Image.open(image_path) as img:
        buffer = io.BytesIO()
        img_format = img.format
        new_width = img.width // 3
        new_height = img.height // 3
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img.save(buffer, format=img_format,quality=60)  # Save in the original format without compression
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    

def select_best_image(keyword, blog_context, candidate_images):
    print("candidate_images",candidate_images)
    prompt = f"""
You are a multimodal AI that evaluates both text and image information. I have a blog article that discusses the concept "{keyword}". Here are the candidate images available:
{candidate_images[0]}

Based on the blog context below, please choose the most relevant image path for the concept "{keyword}".
Blog context:
{blog_context}

Return only the chosen image path as plain text and if you did not find any image relevant to the context return " ".
"""
    response = openai.chat.completions.create(
         model="gpt-4o-mini",  # Replace with the appropriate multimodal model if available
         messages=[
              {"role": "system", "content": "You are a multimodal AI that selects the most relevant image for a blog article based on both visual and textual cues."},
              {"role": "user", "content": prompt}
         ]
    )
    chosen_image = response.choices[0].message.content.strip()

    return chosen_image

def combine_blog_with_llm(blog_draft, mapping):
 
    # Regular expression to detect image placeholders
    pattern = r'\[\[IMAGE:\s*(.*?)\]\]'
    
    def choose_relevant_image(match):
        keyword = match.group(1).strip()
        candidate_images = mapping.get(keyword, [])
        if candidate_images:
            # You could pass a smaller context snippet if desired
            blog_context = blog_draft  
            chosen_image = select_best_image(keyword, blog_context, candidate_images)
            # Return Markdown image syntax for the chosen image
            return f"![{keyword}]({chosen_image})"
        else:
            # If no candidate images exist, leave the placeholder unchanged
            return match.group(0)
    
    refined_blog = re.sub(pattern, choose_relevant_image, blog_draft)
    return refined_blog

def capture_frame_from_youtube(youtube_url, timestamp, output_filename):
    # Ensure the output directory exists
    output_dir = "downloaded_images_youtube_url"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, output_filename)
    

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]/best[ext=mp4]/best',
        'quiet': True,
        'noplaylist': True,
        'cookiefile': 'youtube_cookies.txt',  # Path to your cookies file
    }
    

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            
            # Get the best format URL
            formats = info.get('formats', [])
            formats.sort(key=lambda x: (x.get('height', 0) or 0) * (x.get('width', 0) or 0), reverse=True)
            
            video_url = formats[0]['url'] if formats else None
            
            if not video_url:
                print(f"Failed to extract video URL for {youtube_url}")
                return False
                
            # Open the video stream using OpenCV
            cap = cv2.VideoCapture(video_url)
            
            try:
                if not cap.isOpened():
                    print(f"Failed to open video stream for {youtube_url}")
                    return False
                
                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Normalize timestamp to seconds
                if isinstance(timestamp, str):
                    timestamp = timestamp.replace('s', '')  # Remove the 's' from timestamp string
                    timestamp_seconds = float(timestamp)
                else:
                    timestamp_seconds = float(timestamp)
                    
                # Calculate frame number from timestamp
                frame_number = int(fps * timestamp_seconds)
                
                # Set the video position to the desired timestamp
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                
                # Read the frame
                ret, frame = cap.read()
                
                if ret:
                    # Apply a sharpening filter
                    kernel = np.array([[-1,-1,-1], 
                                      [-1, 9,-1],
                                      [-1,-1,-1]])
                    
                    sharpened_frame = cv2.filter2D(frame, -1, kernel)
                    
                    # Save the frame with high quality
                    cv2.imwrite(output_path, sharpened_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    print(f"Successfully captured frame at {timestamp_seconds}s to {output_path}")
                    return True
                else:
                    print(f"Failed to read frame at timestamp {timestamp_seconds}s")
                    return False
            finally:
                # Always release the capture
                cap.release()
    except Exception as e:
        print(f"Error capturing frame: {str(e)}")
        return False

def find_keyword_timestamps(transcript_data, keywords, url):
    keyword_timestamps = []
    count = 0
    
    # Ensure the download directory exists
    os.makedirs("downloaded_images_youtube_url", exist_ok=True)
    
    for entry in transcript_data:
        for keyword in keywords:
            if keyword.lower() in entry['text'].lower():
                print(f"Found keyword '{keyword}' in: {entry['text']}")
                is_near_existing = False
                
                for _, timestamp, _ in keyword_timestamps:
                    # Normalize timestamps for comparison
                    current_ts = float(entry['timestamp'].replace('s', ''))
                    existing_ts = float(timestamp.replace('s', ''))
                    
                    # If the difference between current and existing timestamp is within the threshold, skip
                    if abs(current_ts - existing_ts) < 10:
                        is_near_existing = True
                        break 
                
                if not is_near_existing:
                    keyword_timestamps.append((keyword, entry['timestamp'], entry['text']))
                    success = capture_frame_from_youtube(url, entry['timestamp'], f"{keyword}_{count}.png")
                    if success:
                        count += 1
                    print(f"Capture {'successful' if success else 'failed'} for {keyword} at {entry['timestamp']}")

def fetch_google_images(query, num_images=1):
    """Fetches image URLs from Google Images using SerpAPI."""
    params = {
        "q": query,                  # Search query
        "tbm": "isch",               # Image search mode
        "num": num_images,           # Number of images
        "api_key": "" # Replace with your SerpAPI key  --------------------------------------------------------
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()
    
    image_urls = [img["original"] for img in results.get("images_results", [])]
    
    return image_urls[0]

def extract_image_keywords(blog_draft,save_dir,save_dir1,formatted_transcript,url,image_to_base64):
    # This regex captures text between '[[IMAGE:' and ']]'
    pattern = r'\[\[IMAGE:\s*(.*?)\]\]'
    keywords = re.findall(pattern, blog_draft)
    save_google_images(keywords, save_dir)
    find_keyword_timestamps(formatted_transcript, keywords,url)
    mapping=process_images_and_create_mapping(keywords, save_dir, save_dir1, image_to_base64)
    blog_draft_with_images = combine_blog_with_llm(blog_draft, mapping)
    return blog_draft_with_images

def extract_video_id(url):
    # Define regex pattern to capture YouTube video ID
    regex_pattern = r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)'
    # Search for the video ID in the URL
    match = re.search(regex_pattern, url)
    # If a match is found, return the video ID
    if match:
        return match.group(1)
    else:
        return None

def generate_blog_draft(transcript, insights):
    prompt = f"""Using the following transcript and extracted insights, generate a well-structured blog post. 
    
    The blog post should include:
    - A catchy, SEO-friendly title
    - An introduction that hooks the reader and explains the video content
    - A main body divided into sections with subheadings, integrating the key themes and takeaways
    
    IMPORTANT INSTRUCTIONS FOR IMAGE PLACEHOLDERS:
    - At appropriate points, include placeholders like [[IMAGE: <keyword>]] where visuals should be inserted
    - Each keyword MUST be an EXACT verbatim phrase that appears in the transcript
    - ** Keywords must be 2-3 words long and should not be a phrase**  and **most important IT should represent a key concept discussed in that section**
    - Do NOT use underscores in keywords
    - Do NOT add words like "diagram", "illustration", or "chart" to keywords
    - Use normal text with spaces between words
    - Before finalizing each keyword, verify it appears word-for-word in the transcript
    - Place each image placeholder precisely where the concept is being explained
    - Use no more than 5 image placeholders total
    
    Transcript:
    {transcript}
    
    Extracted Insights:
    {insights}
    
    Please output the blog post in Markdown format.
    """

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert content creator specialized in creating blog posts from video transcripts. You are extremely careful to follow all instructions precisely, especially regarding image placeholders."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


def extract_blog_insights(transcript):
    prompt = f"""You are a professional blog writer. Analyze the following transcript of a YouTube video and extract the key themes, main arguments, and 2-3 engaging quotes. 
    Your output should include:
    - Core Themes (as a list)
    - Key Takeaways (as bullet points)
    - Engaging Quotes (as a list)
    
    Transcript:
    {transcript}
    """
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a professional blog writer."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

# Function to get transcript with timestamps
def get_video_transcript(video_id):
    try:
        # Fetch transcript for the video
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Format the transcript with timestamps
        formatted_transcript = ""
        main_transcript = ""
        for segment in transcript:
            start_time = segment['start']  # Timestamp (start time of the segment)
            text = segment['text']  # Transcript text
            formatted_transcript += f"[{start_time:.2f}s]: {text}\n"
            main_transcript += text + " "
        

        """Parses each transcript line into a dict with timestamp and text."""
        parsed_entries = []
        for line in formatted_transcript.split('\n'):
            # Expecting lines in the format: "[timestamp]: text"
            match = re.match(r'\[(.*?)\]:\s*(.*)', line)
            if match:
                timestamp, text = match.groups()
                parsed_entries.append({'timestamp': timestamp, 'text': text})

        
        return main_transcript,parsed_entries
    
    except Exception as e:
        return f"Error: {str(e)} - Unable to fetch transcript."



# import tiktoken
# encoding = tiktoken.get_encoding("cl100k_base")
# encode_image_to_base64 = image_to_base64_no_compression("downloaded_images_google/frequency distribution graph.jpg")
# encode_image_to_base64_1=image_to_base64("downloaded_images_google/frequency distribution graph.jpg")
# tokens = encoding.encode(encode_image_to_base64)
# tokens1 = encoding.encode(encode_image_to_base64_1)
# # print(f"Tokenized Output: {tokens}")
# print(f"Number of Tokens: {len(tokens)}")
# print(f"Number of Tokens 1: {len(tokens1)}")

