# standard lib


# third party

# local


def process_items_safely(item_iterator):
    """Processes items from an iterator, handling exceptions individually
       and also handling potential exceptions during iterator creation.
    """

    results = []
    errors = []

    try:
        # Attempt to get the iterator. This is the key change.
        iterator = iter(item_iterator) # Important for generators
    except Exception as e:  # Catch exceptions during iterator creation
        print(f"Error creating iterator: {e}")
        return [], [("Iterator Creation", str(e))] # Return empty results and the iterator creation error

    while True:  # Use a while loop to handle StopIteration
        try:
            item = next(iterator)
            results.append(item)
        except StopIteration:
            break  # Iterator is exhausted
        except ValueError as e:
            print(f"Error processing {item}: {e}")
            errors.append((item, str(e)))
        except TypeError as e:
            print(f"Type Error processing {item}: {e}")
            errors.append((item, str(e)))
        except Exception as e:
            print(f"An unexpected error occurred processing {item}: {e}")
            errors.append((item, str(e)))

    return results, errors
