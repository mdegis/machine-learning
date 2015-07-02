#!/usr/bin/python

def outlierCleaner(predictions, ages, net_worths):
    """
        clean away the 10% of points that have the largest
        residual errors (different between the prediction
        and the actual net worth)

        return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error)
    """
    
    cleaned_data = []
    errors = [(n - p)**2 for n, p in zip(net_worths, predictions)]
    complete_data = zip(ages, net_worths, errors)
    # sort on the 3rd element, error; ascending by default
    complete_data = sorted(complete_data, key=lambda t: t[2])
    # take the first 90%
    cleaned_data = complete_data[:(len(complete_data) * 9) / 10]
    return cleaned_data

