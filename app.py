import streamlit as st
import time


def main():
    st.set_page_config(
        page_title="Project YOLOv10 - AIO2024 Module 01",
        page_icon=":smiley:",
    )
    st.title("Project YOLOv10 - AIO2024 Module 01")
    st.title(':sparkles: :blue[YOLOv10] Helmet Safety Detection Demo')
    # Create progress bar
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)
    st.write("Done!")

if __name__ == "__main__":
    main()
