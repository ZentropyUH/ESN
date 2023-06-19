


from rich.progress import track

def main():
    total = 0
    for value in track(range(100), description="Processing..."):
        # Fake processing time
        total += 1
    print(f"Processed {total} things.")