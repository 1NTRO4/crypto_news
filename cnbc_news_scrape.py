from bs4 import BeautifulSoup
import requests
import csv
import time

# Base URL for the pages to scrape
base_url = "https://www.cnbc.com/cryptoworld/"

# Open a CSV file to write the URLs
file = open('cryptoNews_headlines.csv', "w", newline='')
writer = csv.writer(file)

# Write the header to the CSV file
writer.writerow(["URL"])

page_number = 1
max_pages = 10  # Set this to the maximum number of pages you want to scrape

while True:
    page_to_scrape = requests.get(base_url + f"?page={page_number}")
    soup = BeautifulSoup(page_to_scrape.text, "html.parser")

    # Find all <a> tags with the class 'RiverCard-mediaContainer'
    links = soup.find_all("a", class_="RiverCard-mediaContainer")

    if not links or page_number > max_pages:
        break

    # Extract and write the 'href' attribute from each <a> tag to the CSV file
    for link in links:
        url = link.get('href')
        if url:
            writer.writerow([url])
            print(url)

    page_number += 1
    time.sleep(1)  # Sleep to prevent overwhelming the server (politeness policy)

# Close the CSV file
file.close()