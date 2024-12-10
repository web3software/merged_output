import pandas as pd
import requests
from youtube_transcript_api import YouTubeTranscriptApi
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from googleapiclient.discovery import build
import re
import streamlit as st


# Function to fetch all coins
def get_all_coins(api_key):
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    headers = {"Accepts": "application/json", "X-CMC_PRO_API_KEY": api_key}
    params = {"limit": 100}
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    coins = [{"ticker": coin["symbol"], "name": coin["name"], "slug": coin["slug"]} for coin in data["data"]]
    return pd.DataFrame(coins)


# Function to fetch the transcript for a video
def get_video_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        return transcript
    except Exception as e:
        st.error(f"Error fetching transcript for video {video_id}: {e}")
        return []


def count_coin_mentions_with_names(video_id, transcript, coin_df):
    coin_mentions = {coin: [] for coin in coin_df["name"]}
    for entry in transcript:
        text = entry["text"].lower()
        start_time = entry["start"]
        mentioned_coins = set()

        for _, row in coin_df.iterrows():
            coin_names = [row["name"].lower(), row["slug"].lower()]
            for name in coin_names:
                if re.search(rf'\b{name}\b', text) and name not in mentioned_coins:
                    formatted_time = convert_to_timestamp(start_time)
                    coin_mentions[row["name"]].append(formatted_time)
                    mentioned_coins.add(name)
    return {coin: mentions for coin, mentions in coin_mentions.items() if mentions}

def convert_to_timestamp(start_time):
    hours = int(start_time // 3600)
    minutes = int((start_time % 3600) // 60)
    seconds = int(start_time % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}" if hours > 0 else f"{minutes:02}:{seconds:02}"

# Helper function to create timestamp links
def convert_to_timestamp_link(video_id, start_time):
    hours = int(start_time // 3600)
    minutes = int((start_time % 3600) // 60)
    seconds = int(start_time % 60)
    formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}" if hours > 0 else f"{minutes:02}:{seconds:02}"
    return f'<a href="https://www.youtube.com/watch?v={video_id}&t={int(start_time)}" target="_blank">{formatted_time}</a>'


# Function to fetch video details (for script 1)
def get_video_details(api_key, channel_id, max_results, coin_data):
    youtube = build("youtube", "v3", developerKey=api_key)
    search_request = youtube.search().list(
        part="snippet",
        channelId=channel_id,
        order="date",
        type="video",
        maxResults=max_results
    )
    search_response = search_request.execute()
    video_ids = [item["id"]["videoId"] for item in search_response.get("items", [])]
    video_data = []
    for video_id in video_ids:
        transcript = get_video_transcript(video_id)
        transcript_text = " ".join([entry["text"] for entry in transcript]) if transcript else "Transcript not available."
        coin_mentions = count_coin_mentions_with_names(video_id, transcript, coin_data)
        video_data.append({
            "video_id": video_id,
            "title": next((item["snippet"]["title"] for item in search_response["items"] if item["id"]["videoId"] == video_id), "Unknown Title"),
            "video_link": f"https://www.youtube.com/watch?v={video_id}",
            "transcript": transcript_text,
            "coin_mentions": coin_mentions
        })
    return video_data

def prepare_plot_data(video_data):
    rows = []
    for video in video_data:
        for coin, timestamps in video["coin_mentions"].items():
            if timestamps:
                rows.append({
                    "video_title": video["title"],
                    "video_link": video["video_link"],
                    "coin": coin,
                    "mentions": len(timestamps),
                    "timestamps": ", ".join(timestamps)
                })
    return pd.DataFrame(rows)


# Create plots for visualization (for script 1)
def create_plots(df):
    unique_videos = df["video_title"].unique()
    fig = make_subplots(rows=len(unique_videos), cols=1, shared_xaxes=True, subplot_titles=unique_videos, vertical_spacing=0.1)
    for i, video_title in enumerate(unique_videos):
        video_df = df[df["video_title"] == video_title]
        for coin in video_df["coin"].unique():
            coin_df = video_df[video_df["coin"] == coin]
            fig.add_trace(
                go.Bar(
                    x=coin_df["coin"],
                    y=coin_df["mentions"],
                    name=coin,
                    hovertext=[f"{coin}: {mentions} mentions" for mentions in coin_df["mentions"]],
                    hoverinfo="text",
                ),
                row=i + 1, col=1
            )
    fig.update_layout(
        title="Coin Mentions Across YouTube Videos",
        xaxis_title="Coins",
        yaxis_title="Mentions",
        barmode="stack",
        showlegend=True,
        title_font=dict(size=18),
        legend_title="Coin",
        margin=dict(t=100, b=100),
        height=500 * len(unique_videos)
    )
    return fig


# Get the latest video ID from a YouTube channel (for script 2)
def get_latest_video_from_channel(api_key, channel_id):
    youtube = build("youtube", "v3", developerKey=api_key)
    request = youtube.search().list(
        part="snippet",
        channelId=channel_id,
        maxResults=1,
        order="date"
    )
    response = request.execute()
    if response["items"]:
        video_id = response["items"][0]["id"]["videoId"]
        return video_id
    return None


# Get video comments (for script 2)
def get_video_comments(api_key, video_id):
    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    next_page_token = None
    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token
        )
        response = request.execute()
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
    return comments


# Count mentions of coins in the comments (for script 2)
def count_coin_mentions_in_comments(comments, coin_df):
    coin_mentions = {coin: [] for coin in coin_df["name"]}
    for comment in comments:
        text = comment.lower()
        mentioned_coins = set()
        for _, row in coin_df.iterrows():
            coin_names = [row["name"].lower(), row["slug"].lower()]
            for name in coin_names:
                if re.search(rf'\b{name}\b', text) and name not in mentioned_coins:
                    coin_mentions[row["name"]].append(comment)
                    mentioned_coins.add(name)
    return {coin: mentions for coin, mentions in coin_mentions.items() if mentions}


# Prepare a DataFrame with comments and mentions (for script 2)
def prepare_comments_dataframe(video_id, comments, coin_df):
    coin_mentions = count_coin_mentions_in_comments(comments, coin_df)
    rows = []
    for coin, mentions in coin_mentions.items():
        for mention in mentions:
            rows.append({
                "Video ID": video_id,
                "Coin": coin,
                "Comment": mention
            })
    return pd.DataFrame(rows)


# Create a bar chart for coin mentions (for script 2)
def create_plot(df):
    fig = go.Figure()
    for coin in df["Coin"].unique():
        coin_df = df[df["Coin"] == coin]
        fig.add_trace(
            go.Bar(
                x=coin_df["Coin"],
                y=[len(coin_df)],
                name=coin,
                hovertext=coin_df["Comment"],
                hoverinfo="text",
            )
        )
    fig.update_layout(
        title="Coin Mentions in Comments",
        xaxis_title="Coins",
        yaxis_title="Number of Mentions",
        barmode="stack",
        showlegend=True
    )
    return fig


# Streamlit App
def main():
    st.title("YouTube Coin Mentions Analysis")

    # API keys and channel details
    COINMARKETCAP_API_KEY = st.secrets["COINMARKETCAP_API_KEY"]
    YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
    CHANNEL_ID = st.secrets["CHANNEL_ID"]
    MAX_RESULTS = 5

    # Fetch coin data
    coin_data = get_all_coins(COINMARKETCAP_API_KEY)

    # Section 1: Coin Mentions in Video Transcripts
    st.header("Coin Mentions in Video Transcripts")

    # Get video data and display table
    video_data = get_video_details(YOUTUBE_API_KEY, CHANNEL_ID, MAX_RESULTS, coin_data)
    video_df = prepare_plot_data(video_data)
    st.write("Coin mentions in YouTube video transcripts:")
    st.dataframe(video_df)

    # Plot graph
    if not video_df.empty:
        st.plotly_chart(create_plots(video_df))

    # Section 2: Coin Mentions in Video Comments
    st.header("Coin Mentions in Video Comments")

    # Get the latest video comments
    latest_video_id = get_latest_video_from_channel(YOUTUBE_API_KEY, CHANNEL_ID)
    if latest_video_id:
        comments = get_video_comments(YOUTUBE_API_KEY, latest_video_id)
        comments_df = prepare_comments_dataframe(latest_video_id, comments, coin_data)
        st.write("Coin mentions in YouTube video comments:")
        st.dataframe(comments_df)

        # Plot graph for comments
        if not comments_df.empty:
            st.plotly_chart(create_plot(comments_df))
    else:
        st.error("No videos found in the channel.")

if __name__ == "__main__":
    main()
