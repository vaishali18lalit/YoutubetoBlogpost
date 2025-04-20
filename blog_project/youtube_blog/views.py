# views.py

from django.shortcuts import render
from django.http import JsonResponse
from .youtube_blog_app import extract_video_id ,extract_blog_insights, generate_blog_draft, extract_image_keywords
from .youtube_blog_app import generate_quiz_questions
from .youtube_blog_app import image_to_base64
from .youtube_blog_app import get_video_transcript
import json
import os
from django.conf import settings
from django.shortcuts import render
import shutil
import re

def home(request):
    return render(request, 'home.html')  # Ensure you have a 'home.html' template


save_dir = os.path.join(settings.MEDIA_ROOT, "downloaded_images_google")
save_dir1 = os.path.join(settings.MEDIA_ROOT, "downloaded_images_youtube_url")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(save_dir1, exist_ok=True)


def generate_blog_and_quiz(request):
    if request.method == "POST":
        try:
            video_url = request.POST.get("video_url")
            
            # # Get the option parameter (blog, quiz, or both)
            option = request.POST.get("option", "both")
            print("Option:", option)
            if not video_url:
                return JsonResponse({"error": "YouTube URL not provided."}, status=400)
            
            # # Extract video ID from the URL using the helper function
            video_id = extract_video_id(video_url)
            print("Video ID:", video_id)
            
            if not video_id:
                return JsonResponse({"error": "Invalid YouTube URL."}, status=400)
            
            # # Call the function from youtube_blog_app.py to get the transcript
            main_transcript, formatted_transcript = get_video_transcript(video_id)
            print("Main Transcript:", main_transcript)
            if "Error" in main_transcript:
                return JsonResponse({"error": "Failed to fetch transcript."}, status=400)
            
            blog_content = ""
            quiz_content = ""
            
            # Only generate blog if option is 'blog' or 'both'
            if option == "blog" or option == "quiz":
                
                # Generate blog draft
                insights = extract_blog_insights(main_transcript)
                print("Insights:", insights)
                blog_draft = generate_blog_draft(main_transcript, insights)
                blog_draft_with_images = extract_image_keywords(blog_draft, save_dir, save_dir1, formatted_transcript, video_url, image_to_base64)
                # Extract all image paths from the markdown
                print("Blog Draft with Images:", blog_draft_with_images)
                image_paths = re.findall(r'!\[.*?\]\((.*?)\)', blog_draft_with_images)
                
                # Copy each image to the media directory
                for path in image_paths:
                    # Clean the path if needed
                    clean_path = path.strip().strip("'").strip('"')
                    if os.path.exists(clean_path):
                        # Get just the filename
                        filename = os.path.basename(clean_path)
                        # Destination path in MEDIA_ROOT
                        destination = os.path.join(settings.MEDIA_ROOT, filename)
                        # Copy the file
                        shutil.copy2(clean_path, destination)
                # Define the file path using BASE_DIR
                blog_file_path = os.path.join(settings.BASE_DIR, "blog_draft.md")
                
                with open(blog_file_path, "w") as file:
                    file.write(blog_draft_with_images)
                
                # Read the blog file if it exists
                if os.path.exists(blog_file_path):
                    with open(blog_file_path, "r", encoding="utf-8") as blog_file:
                        blog_content = blog_file.read()
                else:
                    return JsonResponse({"error": "Blog file not found."}, status=404)
            

                # Generate quiz questions
                quiz = generate_quiz_questions(main_transcript)
                
                # Define the file path using BASE_DIR
                quiz_file_path = os.path.join(settings.BASE_DIR, "quiz.md")
                
                with open(quiz_file_path, "w") as file:
                    file.write(quiz)
                
                # Read the quiz file if it exists
                if os.path.exists(quiz_file_path):
                    with open(quiz_file_path, "r", encoding="utf-8") as quiz_file:
                        quiz_content = quiz_file.read()
                else:
                    return JsonResponse({"error": "Quiz file not found."}, status=404)
            
            # Prepare the response
            response_data = {
                "message": "Content generated successfully.",
                "blog_content": blog_content,
                "quiz_content": quiz_content
            }
            
            print("Response Data:", json.dumps(response_data, indent=4))
            return JsonResponse(response_data)
                
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    
    return render(request, "index.html")  # Serve the form page for GET requests