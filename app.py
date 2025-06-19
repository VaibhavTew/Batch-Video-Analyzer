import os
os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "400"

import os
import tempfile
import threading
import shutil
from dotenv import load_dotenv
load_dotenv()
import sqlite3
import json
from pathlib import Path
# ── FFmpeg Workaround ────────────────────────────────────────────────────────
import imageio_ffmpeg as ffmpeg
# Tell MoviePy exactly which ffmpeg binary to use
import moviepy.config as mpy_cfg
ffmpeg_path = ffmpeg.get_ffmpeg_exe()
os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path
# mpy_cfg.change_settings({"FFMPEG_BINARY": ffmpeg_path})  # Removed: not a valid attribute
os.environ["FFMPEG_BINARY"] = ffmpeg_path  # Set for moviepy compatibility
# ── Core Imports ─────────────────────────────────────────────────────────────
import streamlit as st
from streamlit_option_menu import option_menu
from openai import AzureOpenAI
from moviepy.editor import VideoFileClip
from azure.cognitiveservices.speech import (
    SpeechConfig, SpeechRecognizer, AudioConfig, ResultReason
)
import pandas as pd
import re
# ── Page Config & CSS ────────────────────────────────────────────────────────
 # ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Interview Analysis")
 
 # ── SQLite Initialization ──────────────────────────────────────────────────
DB_PATH = Path("projects.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.execute("""
     CREATE TABLE IF NOT EXISTS projects (
         id INTEGER PRIMARY KEY AUTOINCREMENT,
         name TEXT UNIQUE,
         objective TEXT,
         type TEXT,
         description TEXT,
         video_path TEXT,
         guide_path TEXT,
         extras TEXT
     )
 """)
conn.commit()
st.markdown("""
<style>
    /* Remove any default Streamlit tab border */
    [data-baseweb="tab-list"] {
      border: none !important;
      position: relative;        /* for our custom underline */
      margin-bottom: 16px;
    }

    /* Draw only our purple underline via ::after */
    [data-baseweb="tab-list"]::after {
      content: "";
      position: absolute;
      bottom: 0;
      left: 0;
      width: 100%;
      height: 2px;
      background-color: #6200EE;
    }

    /* Global */
    .main-header    { color: #6200EE; font-size: 24px; font-weight: bold; margin-bottom: 20px; }
    .sub-header     { color: #6200EE; font-size: 18px; font-weight: bold; margin-bottom: 15px; }
    .section-header { font-size: 18px; font-weight: bold; margin-top: 15px; margin-bottom: 10px; }
    .video-grid     { display: flex; flex-wrap: wrap; gap: 20px; margin-top: 10px; }
    .video-card     { border: 1px solid #ddd; border-radius: 8px; padding: 15px; width: calc(33% - 20px); }
    .video-info     { margin-bottom: 10px; }
    .button-container { display: flex; gap: 10px; }
    .button-white   { flex: 1; background-color: white; border: 1px solid #ddd; color: black; padding: 8px; text-align: center; border-radius: 4px; }
    .button-purple  { flex: 1; background-color: #6200EE; color: white; padding: 8px; text-align: center, border-radius: 4px; }
    .small-btn button {padding: 4px 8px !important;font-size: 0.9rem !important;}
                                   
    /* ── Topics Table Styling ───────────────────────────────────────────── */
   .topics-table {width: 100%;border-collapse: collapse;margin-top: 12px;}

  .topics-table th,
  .topics-table td {
  border: 1px solid #ADD8E6;   /* light-blue border */
  padding: 8px;
  background-color: white;     /* data rows stay white */
  }

  .topics-table th {
  background-color: #E3F2FD;   /* light-blue header fill */
  text-align: left;
  }
        
    /* ── Custom st.tabs (pill-shaped) ───────────────────────────────── */
    [data-baseweb="tab-list"] button {
      background-color: white !important;
      color: black !important;
      border: 1px solid #ddd !important;
      border-radius: 999px !important;
      padding: 6px 16px !important;
      margin-right: 8px !important;
      font-weight: 500;
    }
    [data-baseweb="tab-list"] button[aria-selected="true"] {
      background-color: #6200EE !important;
      color: white !important;
      border-color: #6200EE !important;
    }
    [data-baseweb="tab-list"] button:focus {
      box-shadow: none !important;
    }

    /* Hide default Streamlit elements */
    #MainMenu, footer { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)
# ── Azure + OpenAI Client ────────────────────────────────────────────────────
AZ_SPEECH_KEY     = os.getenv("AZURE_SPEECH_KEY")
AZ_SPEECH_REGION  = os.getenv("AZURE_SPEECH_REGION")
AZ_OAI_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT")
AZ_OAI_KEY        = os.getenv("AZURE_OPENAI_API_KEY")
OAI_API_VER       = os.getenv("OPENAI_API_VERSION","2023-03-15-preview")
OAI_DEPLOY        = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

client = AzureOpenAI(
    api_key=AZ_OAI_KEY,
    api_version=OAI_API_VER,
    azure_endpoint=AZ_OAI_ENDPOINT
)


import re

# 1) Introduction patterns (English only, first-name)
INTRO_PATTERNS = [
    r"\bMy name is\s+\"?(?P<name>[A-Z][a-z]+)\"?",
    r"\bI am\s+\"?(?P<name>[A-Z][a-z]+)\"?",
    r"\bI'm\s+\"?(?P<name>[A-Z][a-z]+)\"?",
    r"\bThis is\s+\"?(?P<name>[A-Z][a-z]+)\"?",
    r"\bYou can call me\s+\"?(?P<name>[A-Z][a-z]+)\"?"
]

def find_intro_speakers(transcript: str, max_lines: int = 10) -> set:
    detected = set()
    for ln in transcript.splitlines()[:max_lines]:
        if ":" not in ln:
            continue
        _, utterance = ln.split(":", 1)
        for pat in INTRO_PATTERNS:
            m = re.search(pat, utterance, flags=re.IGNORECASE)
            if m and m.group("name"):
                detected.add(m.group("name"))
    return detected


# ── Helper Functions ─────────────────────────────────────────────────────────
def extract_audio(video_path, wav_path):
    clip = VideoFileClip(video_path)
    try:
        if clip.audio:
            clip.audio.write_audiofile(wav_path, codec="pcm_s16le")
        else:
            # No audio track: write a tiny silent WAV so transcription returns empty
            import wave
            with wave.open(wav_path, 'w') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b'')  # zero‐length data
    finally:
        clip.close()


def extract_clips(text):
    prompt = (
        "From the transcript below, output a single-line timeline: "
        "timestamp and brief note for each minute.\n"
        f"Transcript:\n'''{text}'''"
    )
    resp = client.chat.completions.create(
        model=OAI_DEPLOY,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return [
        line.strip()
        for line in resp.choices[0].message.content.split("\n")
        if line.strip()
    ]

def extract_topics(text):
    prompt = (
        "From the transcript below, perform topic modeling: list each topic with a brief summary.\n"
        f"Transcript:\n'''{text}'''"
    )
    resp = client.chat.completions.create(
        model=OAI_DEPLOY,
        messages=[{"role":"user","content":prompt}],
        max_tokens=800
    )
    # Get non-empty lines
    lines = [
        ln.strip()
        for ln in resp.choices[0].message.content.splitlines()
        if ln.strip()
    ]

    rows = []
    # We expect pairs: "Topic X: Name" then "Summary: description"
    for i in range(0, len(lines), 2):
        topic_line   = lines[i]
        summary_line = lines[i+1] if i+1 < len(lines) else ""
        # Grab text after the first colon
        topic   = topic_line.split(":",1)[1].strip()   if ":" in topic_line   else topic_line
        summary = summary_line.split(":",1)[1].strip() if ":" in summary_line else summary_line
        rows.append({"Topic": topic, "Summary": summary})
    return rows


def transcribe_audio(wav_path):
    cfg   = SpeechConfig(AZ_SPEECH_KEY, AZ_SPEECH_REGION)
    audio = AudioConfig(filename=wav_path)
    rec   = SpeechRecognizer(speech_config=cfg, audio_config=audio)
    parts, done = [], threading.Event()
    rec.recognized.connect(
        lambda evt: parts.append(evt.result.text)
        if evt.result.reason == ResultReason.RecognizedSpeech else None
    )
    rec.session_stopped.connect(lambda evt: done.set())
    rec.canceled.connect(lambda evt: done.set())
    rec.start_continuous_recognition()
    done.wait()
    rec.stop_continuous_recognition()
    return " ".join(parts)

def summarize(text):
    prompt = (
        "You are an analytics assistant. Return 3-5 concise bullet points of the transcript.\n"
        f"Transcript:\n'''{text}'''"
    )
    resp = client.chat.completions.create(
        model=OAI_DEPLOY,
        messages=[{"role":"user","content":prompt}],
        max_tokens=500
    )
    return [line.strip() for line in resp.choices[0].message.content.split("\n") if line.strip()]

# Add a new helper function to extract speaker names via OpenAI
def extract_speakers(text):
    prompt = (
        "Extract a list of distinct speaker names from the transcript. "
        "Return a comma-separated list of first names only."
    )
    resp = client.chat.completions.create(
         model=OAI_DEPLOY,
         messages=[{"role": "user", "content": f"{prompt}\nTranscript:\n'''{text}'''"}],
         max_tokens=100
    )
    names = resp.choices[0].message.content.strip().split(',')
    return [n.strip() for n in names if n.strip()]

# ── Session State Defaults ──────────────────────────────────────────────────
if "projects" not in st.session_state:
     cur = conn.execute("SELECT id,name,objective,type,description,video_path,guide_path,extras FROM projects")
     st.session_state["projects"] = [
         {
             "id": r[0],
             "name": r[1],
             "objective": r[2],
             "type": r[3],
             "description": r[4],
             "video": r[5],
             "guide": r[6],
             "extras": json.loads(r[7] or "{}")
         }
         for r in cur.fetchall()
     ]
if "adding" not in st.session_state:
    st.session_state["adding"] = False
if "selected_speaker" not in st.session_state:
    st.session_state["selected_speaker"] = None

# ── Sidebar & Navigation ────────────────────────────────────────────────────
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Analysis", "Create New Project", "Unified Insights"],
        default_index=0 if not st.session_state.adding else 1,
        icons=["bar-chart", "plus-circle", "layers"],
        orientation="vertical",
        styles={"nav-link-selected":{"background-color":"#6200EE","color":"white"}}
    )
    st.session_state.adding = (selected == "Create New Project")
    st.session_state.unified = (selected == "Unified Insights")
        # ── Project List Dropdown ───────────────────────────────────────────────
    st.markdown("### Select Project")
    names = [p["name"] for p in st.session_state.projects]
    pick = st.selectbox("", [""] + names, label_visibility="collapsed", key="proj_selector")
    if pick:
         # 1) Switch project
        proj = next(p for p in st.session_state.projects if p["name"] == pick)
        st.session_state.project = proj
 
         # 2) Clear previous project's state
        for key in ["transcript", "topics", "clips", "clips_dir"]:
            if key in st.session_state:
                del st.session_state[key]
 
         # 3) Load new project's artifacts if they exist
        extras = proj.get("extras", {})
 
         # Transcript
        tpath = extras.get("transcript_path")
        if tpath and Path(tpath).is_file():
            with open(tpath, "r") as f:
                st.session_state.transcript = f.read()
 
         # Topics (we'll load the JSON when needed in the Topics tab)
        tjson = extras.get("topics_path")
        if tjson and Path(tjson).is_file():
            st.session_state.topics = tjson
 
         # Clips list
        cdir = extras.get("clips_dir")
        if cdir and Path(cdir).is_dir():
             # build a list of clip infos for display
            clips = []
            for fname in sorted(Path(cdir).glob("clip_*.mp4")):
                 # parse start/end from filename
                 parts = fname.stem.split("_")
                 start, end = int(parts[1]), int(parts[2])
                 clips.append({"start": start, "end": end, "path": str(fname)})
            st.session_state.clips = clips
            st.session_state.clips_dir = cdir

# ── Unified Insights Flow ───────────────────────────────────────────────────
if st.session_state.unified:
    st.markdown('<div class="main-header">Unified Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload up to 4 videos</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["mp4","mov"], accept_multiple_files=True, key="uni_vids")
    if uploaded and len(uploaded) > 4:
        st.error("Please upload no more than 4 videos.")
    if st.button("Process Unified Insights") and uploaded:
        transcripts = []
        # Transcribe each video
        for idx, vid in enumerate(uploaded, start=1):
            tmp_mp4 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            open(tmp_mp4, "wb").write(vid.getbuffer())
            wav = tmp_mp4.replace('.mp4','.wav')
            extract_audio(tmp_mp4, wav)
            txt = transcribe_audio(wav)
            os.remove(tmp_mp4); os.remove(wav)
            transcripts.append((idx, txt))
        # Build combined transcript with subheadings
        full = []
        for idx, txt in transcripts:
            full.append(f"### Video {idx} Transcript")
            full.extend(txt.splitlines())
        full_text = "\n".join(full)
        # Display full transcript
        st.markdown('<div class="section-header">Full Transcript</div>', unsafe_allow_html=True)
        st.text_area("", full_text, height=400)
        # Unified Insights summary
        st.markdown('<div class="section-header">Unified Insights</div>', unsafe_allow_html=True)
        bullets = summarize(full_text)
        for b in bullets:
            st.write(f"- {b}")
        # Chatbot under Unified Insights
        st.markdown('---')
        st.markdown('<div class="section-header">Chatbot: Ask about the unified transcript</div>', unsafe_allow_html=True)
        q_uni = st.text_input("Your question:", key="uni_q")
        if st.button("Ask!", key="uni_ask"):
            if not full_text:
                st.error("Please process transcripts first.")
            else:
                ans = client.chat.completions.create(
                    model=OAI_DEPLOY,
                    messages=[
                        {"role":"system","content":"Answer from unified transcript."},
                        {"role":"user","content":f"Context:'''{full_text}'''\n\nQuestion: {q_uni}"}
                    ],
                    max_tokens=200
                ).choices[0].message.content
                st.write(f"**Answer:** {ans}")
    elif not uploaded:
        st.info("Upload videos above to begin.")
        

# ── CREATE NEW PROJECT SCREEN ────────────────────────────────────────────────
if st.session_state.adding:
    # User profile
    st.markdown("""
      <div style="display:flex;justify-content:flex-end;align-items:center;">
        <div style="margin-right:10px;"></div>
        <div style="width:60px;height:60px;border-radius:70%;background-color:#6200EE;color:white;
                    display:flex;align-items:center;justify-content:center;font-weight:bold;">
          C5I
        </div>
      </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">Interview Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Create New Project</div>', unsafe_allow_html=True)

    project_name       = st.text_input("Project Name", placeholder="Enter Project Name")
    project_objective  = st.text_input("Project Objective", placeholder="Enter Project Objective")
    video_type         = st.selectbox(
        "Type of Video ( Single Participant / Multiple Participants )",
        ["Single Participant", "Multiple Participants"]
    )
    project_description = st.text_area(
        "Project Description", placeholder="Enter Project Description", height=120
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Upload Video Files</div>', unsafe_allow_html=True)

    upl_cols = st.columns(2)
    with upl_cols[0]:
        st.markdown("""
          <div class="upload-area">
            <div class="upload-icon">🎥</div>
            <div class="upload-text">Upload Videos</div>
            <div class="upload-subtext">.mp4, .mov ( Max 25 MB/file )</div>
            """ + (
              "" if not (uploaded := st.file_uploader("", type=["mp4","mov"], key="vid")) 
              else f"<div class='file-box'>{uploaded.name}</div>"
            ) + """
          </div>
        """, unsafe_allow_html=True)

    with upl_cols[1]:
        st.markdown("""
          <div class="upload-area">
            <div class="upload-icon">📄</div>
            <div class="upload-text">Upload Discussion Guide</div>
            <div class="upload-subtext">.doc, .pdf ( Max 25 MB/file )</div>
            """ + (
              "" if not (guide := st.file_uploader("", type=["doc","pdf"], key="guide")) 
              else f"<div class='file-box'>{guide.name}</div>"
            ) + """
          </div>
        """, unsafe_allow_html=True)

    if st.button("Create Project"):
        if not (project_name and project_objective and project_description and uploaded):
            st.error("Please complete all fields and upload a video.")
        else:
            # Save project folder
            base = Path("saved_projects") / project_name
            base.mkdir(parents=True, exist_ok=True)
            vid_path = base / uploaded.name
            with open(vid_path, "wb") as f: f.write(uploaded.getbuffer())
            guide_path = None
            if guide:
                guide_path = base / guide.name
                with open(guide_path, "wb") as f: f.write(guide.getbuffer())
 
             # Insert into SQLite with empty extras
            cur = conn.execute(
                 "INSERT INTO projects (name,objective,type,description,video_path,guide_path,extras) VALUES (?,?,?,?,?,?,?)",
                 (project_name, project_objective, video_type, project_description,
                  str(vid_path), str(guide_path) if guide else None, json.dumps({}))
             )
            conn.commit()
            pid = cur.lastrowid
 
            new_proj = {
                 "id": pid,
                 "name": project_name,
                 "objective": project_objective,
                 "type": video_type,
                 "description": project_description,
                 "video": str(vid_path),
                 "guide": str(guide_path) if guide else None,
                 "extras": {}
             }
            st.session_state.projects.append(new_proj)
            st.session_state.project = new_proj
            st.session_state.adding = False
            st.success("Project created! 🎉")

# ── ANALYSIS SCREEN with Tabs ───────────────────────────────────────────────
else:
    if "project" not in st.session_state:
        st.info("No project selected. Switch to 'Create New Project' to get started.")
        st.stop()

    # Ensure transcript auto-loads (if a transcript file exists) before using tabs
    p = st.session_state.project
    transcript = st.session_state.get("transcript", "")

    # Ensure transcript is available for the selected project
    p = st.session_state.project
    transcript = st.session_state.get("transcript", "")

    if not transcript:
        st.info("Processing transcript, please wait...")
        # Generate a transcript file path in the same project folder
        tpath = Path(p["video"]).parent / "transcript.txt"
        # Create a temporary WAV file from the video
        wav_path = str(p["video"]).replace(".mp4", ".wav")
        extract_audio(p["video"], wav_path)
        # Transcribe the audio
        transcript = transcribe_audio(wav_path)
        os.remove(wav_path)
        # Save transcript to file and update extras
        with open(tpath, "w") as f:
            f.write(transcript)
        p["extras"]["transcript_path"] = str(tpath)
        conn.execute("UPDATE projects SET extras=? WHERE id=?", (json.dumps(p["extras"]), p["id"]))
        conn.commit()
        st.session_state.transcript = transcript

    # Define tabs before using them
    tabs = st.tabs(["Insights", "Topics", "Video Clips"])

    with tabs[0]:
        # Auto-populate Transcript and Summary
        st.markdown("<div class='main-header'>Interview Analysis</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='user-section'>{p['name']}</div>", unsafe_allow_html=True)
        
        # Show video and filters
        vid_col, filt_col = st.columns([3, 3])
        with vid_col:
            st.video(p["video"], format="video/mp4")
        with filt_col:
            if transcript:
                # Populate speakers using OpenAI extraction
                speakers = extract_speakers(transcript)
                sel_speaker = st.selectbox("Choose Speaker", ["All"] + speakers)
                # Populate topics dropdown from stored topics file
                proj_dir = Path(p["video"]).parent
                topics_path = proj_dir / "topics.json"
                if topics_path.exists():
                    with open(topics_path, "r") as f:
                        topics_rows = json.load(f)
                    topics_list = [row["Topic"] for row in topics_rows]
                else:
                    topics_list = []
                sel_topic = st.selectbox("Choose Topic", ["All"] + topics_list)
            else:
                st.markdown("<div style='height:80px;'></div>", unsafe_allow_html=True)

        # Auto-generate or load timeline & insights JSON once transcript exists
        if transcript:
            proj_dir = Path(p["video"]).parent
            timeline_path = proj_dir / "timeline.json"
            insights_path = proj_dir / "insights.json"

            if not timeline_path.exists():
                timeline_data = extract_clips(transcript)
                with open(timeline_path, "w") as f:
                    json.dump(timeline_data, f)
                p["extras"]["timeline_path"] = str(timeline_path)
                conn.execute("UPDATE projects SET extras=? WHERE id=?", (json.dumps(p["extras"]), p["id"]))
                conn.commit()
            else:
                with open(timeline_path, "r") as f:
                    timeline_data = json.load(f)

            if not insights_path.exists():
                insights_data = summarize(transcript)
                with open(insights_path, "w") as f:
                    json.dump(insights_data, f)
                p["extras"]["insights_path"] = str(insights_path)
                conn.execute("UPDATE projects SET extras=? WHERE id=?", (json.dumps(p["extras"]), p["id"]))
                conn.commit()
            else:
                with open(insights_path, "r") as f:
                    insights_data = json.load(f)

            col_tl, col_div, col_ins = st.columns([20, 1, 20])
            with col_tl:
                st.markdown("<div class='section-header'>Timeline</div>", unsafe_allow_html=True)
                for line in timeline_data:
                    st.write(line)
            with col_div:
                st.markdown(
                    """
                    <div style="
                      height: 500px;
                      border-left: 1px solid #ccc;
                      margin: 0px 10px;
                    "></div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_ins:
                st.markdown("<div class='section-header'>Insights</div>", unsafe_allow_html=True)
                for bullet in insights_data:
                    st.write(f"- {bullet}")

        if transcript:
            filtered_lines = []
            for ln in transcript.splitlines():
                # (Filtering logic here)
                filtered_lines.append(ln)
            st.markdown('<div class="section-header">Transcript</div>', unsafe_allow_html=True)
            st.text_area("", "\n".join(filtered_lines), height=300)

    with tabs[1]:
      st.markdown('<div class="section-header">Topic Modeling</div>', unsafe_allow_html=True)
      if transcript:
        proj = st.session_state.project
        proj_dir = Path(proj["video"]).parent
        topics_path = proj_dir / "topics.json"

        # Topics generation / load:
        if not topics_path.exists():
            rows = extract_topics(transcript)
            with open(topics_path, "w") as f:
                json.dump(rows, f)
            proj["extras"]["topics_path"] = str(topics_path)
            conn.execute("UPDATE projects SET extras=? WHERE id=?",
                         (json.dumps(proj["extras"]), proj["id"]))
            conn.commit()
        else:
            with open(topics_path, "r") as f:
                rows = json.load(f)

        # Build HTML table per mockup
        html = '<table class="topics-table"><tr><th>Topic</th><th>Summary</th></tr>'
        for r in rows:
            html += f"<tr><td>{r['Topic']}</td><td>{r['Summary']}</td></tr>"
        html += "</table>"
        st.markdown(html, unsafe_allow_html=True)
      else:
        st.info("Transcript not available yet.")
            
    # ─ Updated Video Clips tab ────────────────────────────────────────────────
        # ─ Updated Video Clips tab ────────────────────────────────────────────────
    # Revised Video Clips section to ensure all clips generate, even without audio
# and to handle proper cleanup after processing all clips.
        # ─ Updated Video Clips tab ────────────────────────────────────────────────
    with tabs[2]:
        st.markdown('<div class="section-header">Video Clips</div>', unsafe_allow_html=True)

        proj = st.session_state.project
        proj_clips_dir = Path(proj["video"]).parent / "clips"

        # Auto-load clips if present in the saved project folder
        existing_clips = list(proj_clips_dir.glob("clip_*.mp4")) if proj_clips_dir.exists() else []
        if existing_clips:
            clips = []
            for fname in sorted(existing_clips):
                parts = fname.stem.split("_")
                if len(parts) >= 3:
                    start, end = int(parts[1]), int(parts[2])
                    clip_info = {"start": start, "end": end, "path": str(fname)}
                    cache_file = fname.with_suffix(".json")
                    if cache_file.exists():
                        with open(cache_file, "r") as cf:
                            cache_data = json.load(cf)
                        clip_info["transcript"] = cache_data.get("transcript", "")
                        clip_info["key_insights"] = cache_data.get("key_insights", [])
                    else:
                        # Generate on the fly
                        wav_path = str(fname).replace(".mp4", ".wav")
                        extract_audio(str(fname), wav_path)
                        clip_transcript = transcribe_audio(wav_path)
                        os.remove(wav_path)
                        clip_summary = summarize(clip_transcript)
                        clip_info["transcript"] = clip_transcript
                        clip_info["key_insights"] = clip_summary
                        # Save cache for future use
                        with open(fname.with_suffix(".json"), "w") as cf:
                            json.dump({"transcript": clip_transcript, "key_insights": clip_summary}, cf)
                clips.append(clip_info)
            st.session_state["clips"] = clips
            st.info(f"Loaded {len(clips)} saved clip(s).")
        else:
            st.session_state["clips"] = []
            st.info("No clips found in saved projects.")

        # Button to (re)generate clips if needed
        if st.button("Auto Clip per Minute"):
            if proj_clips_dir.exists():
                shutil.rmtree(proj_clips_dir)
            proj_clips_dir.mkdir(parents=True, exist_ok=True)
            tmp_mp4 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            shutil.copy(str(proj["video"]), tmp_mp4)
    
            try:
                clip_full = VideoFileClip(tmp_mp4)
            except Exception as e:
                st.error(f"Failed to load video for clipping:\n{e}")
                os.remove(tmp_mp4)
                st.stop()
    
            duration = int(clip_full.duration)
            new_clips = []
    
            for start_sec in range(0, duration, 60):
                end_sec = min(start_sec + 60, duration)
                out_path = proj_clips_dir / f"clip_{start_sec}_{end_sec}.mp4"
    
                subclip = None
                try:
                    subclip = clip_full.subclip(start_sec, end_sec)
                    kwargs = {"codec": "libx264", "threads": 2}
                    if subclip.audio:
                        kwargs.update(audio_codec="aac", bitrate="4000k")
                    else:
                        kwargs["audio"] = False
                    subclip.write_videofile(str(out_path), **kwargs)
                    # After writing the clip, generate its transcript and key insights.
                    wav_path = str(out_path).replace(".mp4", ".wav")
                    extract_audio(str(out_path), wav_path)
                    clip_transcript = transcribe_audio(wav_path)
                    os.remove(wav_path)
                    clip_summary = summarize(clip_transcript)
                    # Cache these details in a JSON file alongside the clip.
                    cache_path = out_path.with_suffix(".json")
                    with open(cache_path, "w") as cf:
                        json.dump({"transcript": clip_transcript, "key_insights": clip_summary}, cf)
                    new_clips.append({"start": start_sec, "end": end_sec, "path": str(out_path),
                                      "transcript": clip_transcript, "key_insights": clip_summary})
                except Exception as e:
                    st.warning(f"Could not process clip {start_sec}-{end_sec}s: {e}")
                finally:
                    if subclip:
                        subclip.close()
    
            clip_full.close()
            os.remove(tmp_mp4)
    
            st.session_state["clips"] = new_clips
            proj["extras"]["clips_dir"] = str(proj_clips_dir)
            conn.execute("UPDATE projects SET extras=? WHERE id=?", (json.dumps(proj["extras"]), proj["id"]))
            conn.commit()
            st.success(f"Created {len(new_clips)} clip(s).")
    
        clips = st.session_state.get("clips", [])
        if not clips:
            st.info("No clips available. Click **Auto Clip per Minute** to generate clips.")
        else:
            cols_per_row = 3
            for idx, clip_info in enumerate(clips):
                if idx % cols_per_row == 0:
                    row_cols = st.columns(cols_per_row, gap="large")
                col = row_cols[idx % cols_per_row]
                with col:
                    st.markdown(
                        f"""
                        <div style="
                          border: 1px solid #ddd;
                          border-radius: 8px;
                          box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
                          padding: 12px;
                          margin-bottom: 16px;
                        ">
                        <strong>Clip {idx+1}: {clip_info['start']}s–{clip_info['end']}s</strong>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.video(clip_info["path"], format="video/mp4")
                    st.markdown("**Transcript:**")
                    st.text_area("", clip_info["transcript"], height=150)
                    st.markdown("**Key Insights:**")
                    for bullet in clip_info["key_insights"]:
                        st.write(f"- {bullet}")
            if (idx + 1) % cols_per_row == 0:
                st.markdown("---")

# End of tabs – Chatbot section remains outside and unchanged
st.markdown("---")
st.markdown('<div class="section-header">Chatbot: Ask about the video</div>', unsafe_allow_html=True)
q = st.text_input("Your question:")
if st.button("Ask!"):
    if not transcript:
        st.error("Please transcribe first.")
    else:
        ans = client.chat.completions.create(
            model=OAI_DEPLOY,
            messages=[
                {"role": "system", "content": "Answer from transcript."},
                {"role": "user", "content": f"Context:\n'''{transcript}'''\nQuestion: {q}"}
            ],
            max_tokens=200
        ).choices[0].message.content
        st.write(f"**Answer:** {ans}")
