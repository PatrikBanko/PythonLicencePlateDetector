import csv, os


def levenshtein_distance(word1, word2):
    # Initialize a matrix to store distances
    distances = [[0] * (len(word2) + 1) for _ in range(len(word1) + 1)]

    # Initialize the first row and column
    for i in range(len(word1) + 1):
        distances[i][0] = i
    for j in range(len(word2) + 1):
        distances[0][j] = j

    # Fill in the matrix
    for i in range(1, len(word1) + 1):
        for j in range(1, len(word2) + 1):
            if word1[i - 1] == word2[j - 1]:
                substitution_cost = 0
            else:
                substitution_cost = 1
            distances[i][j] = min(
                distances[i - 1][j] + 1,
                distances[i][j - 1] + 1,
                distances[i - 1][j - 1] + substitution_cost,
            )

    # Return the distance between the last characters
    return distances[-1][-1]


def compare_columns(csv_file):
    matches = 0
    total_rows = 0

    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row if it exists

        for row in reader:
            if len(row) == 3:  # Ensure there are at least 3 columns
                total_rows += 1
                # if row[1] == row[2]:  # Compare values in second and third columns
                if levenshtein_distance(row[1], row[2]) <= 1:
                    matches += 1

    return matches, total_rows


def generate_report(matches, total_rows):
    images_directory = "mini_test_1/"
    files = os.listdir(images_directory)
    # Count the number of files
    num_files = len(files)
    print(f"\nBroj datoteka na testu: {num_files}")

    read_plates = "./extracted_plates"
    files_read = os.listdir(read_plates)
    # Count the number of files
    num_files = len(files_read)
    print(f"Broj procitanih tablica: {num_files}")

    print(f"\nUkupno podudaranja: {matches}")
    print(f"Ukupan broj provjerenih tablica: {total_rows}")

    print(f"\nPostotak podudaranja: {matches / total_rows * 100:.2f}%\n")


csv_file = "image_list_predicted.csv"
matches, total_rows = compare_columns(csv_file)
generate_report(matches, total_rows)
