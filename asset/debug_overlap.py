import pandas as pd
"""
For debug purpose, check if there is any overlap between spawned objects
"""
def check_overlaps_corrected(data):
    """
    Identify and print overlaps between objects in a DataFrame based on their coordinates and dimensions.

    The function groups the data by a unique identifier ('id'), sorts each group by longitude ('long') and latitude ('lat'),
    and checks for overlaps between adjacent objects within each group. An overlap is identified if the geographical bounds
    (defined by 'long', 'lat', 'len', and 'width') of one object intersect with those of the next object in the sorted group.

    Parameters:
    - data (pandas.DataFrame): A DataFrame containing the object data with columns 'id', 'long', 'lat', 'len', and 'width'.

    Returns:
    - None: The function prints the details of overlapping objects and does not return anything.

    Output:
    - Prints the details of each pair of overlapping objects, including their 'id', 'long', 'lat', 'len', and 'width'.
    """
    # Group data by id
    grouped_data = data.groupby('id')

    # Iterate through each group and check for overlaps
    overlaps = []
    for id, group in grouped_data:
        # Sort the group by longitude and latitude
        sorted_group = group.sort_values(by=['long', 'lat'])

        # Iterate through the sorted objects to check for overlaps
        for i in range(len(sorted_group) - 1):
            current_obj = sorted_group.iloc[i]
            next_obj = sorted_group.iloc[i + 1]

            # Check if the next object is within the bounds of the current object
            if (next_obj['long'] < current_obj['long'] + current_obj['len']) and \
               (next_obj['lat'] < current_obj['lat'] + current_obj['width']):
                print("!!!!Overlaps Founded")
                print(current_obj)
                print(next_obj)

csv_file_path = 'D:\\research\\metavqa_main\\MetaVQA\\asset\\spawned_objects_log.csv'  # Replace with your CSV file path
data = pd.read_csv(csv_file_path)

# Check for overlaps
overlaps = check_overlaps_corrected(data)
