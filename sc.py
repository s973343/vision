from scenedetect import detect, ContentDetector

scene_list = detect(r'.\data\Trishul_480P.mp4', ContentDetector())

for i, (start, end) in enumerate(scene_list, 1):
    duration = end - start   # Timecode subtraction
    print(f"Scene {i}:")
    print(f"  Start   : {start}")
    print(f"  End     : {end}")
    print(f"  Duration: {duration}\n")


