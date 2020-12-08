

#Chapter 1 - Exploring Linear Trends





#--------------------------------------------------------------#

# Reasons for Modelling : Interpolation
# Compute the total change in distance and change in time

total_distance = distances[-1] - distances[0]
total_time = times[-1] - times[0]

# Estimate the slope of the data from the ratio of the changes

average_speed = total_distance / total_time

# Predict the distance traveled for a time not measured

elapse_time = 2.5
distance_traveled = average_speed * elapse_time
print("The distance traveled is {}".format(distance_traveled))

#---------------------------------------------------------------#

# Reasons for Modelling : Extrapolation
# Select a time not measured.

time = 8

# Use the model to compute a predicted distance for that time.

distance = model(time)

# Inspect the value of the predicted distance traveled.

print(distance)

# Determine if you will make it without refueling.

answer = (distance <= 400)
print(answer)

#---------------------------------------------------------------#

# Reasons for Modelling : Estimating Relationships
# Complete the function to model the efficiency.

def efficiency_model(miles, gallons):
   return np.mean( miles / gallons )

# Use the function to estimate the efficiency for each car.

car1['mpg'] = efficiency_model(car1['miles'] , car1['gallons'] )
car2['mpg'] = efficiency_model(car2['miles'] , car2['gallons'] )

# Finish the logic statement to compare the car efficiencies.

if car1['mpg'] > car2['mpg'] :
    print('car1 is the best')
elif car1['mpg'] < car2['mpg'] :
    print('car2 is the best')
else:
    print('the cars have the same efficiency')

#-------------------------------------------------------------------#

# Visualizing Linear Relationships
#Plotting the Data

# Create figure and axis objects using subplots()
fig, axis = plt.subplots()

# Plot line using the axis.plot() method
line = axis.plot(times , distances , linestyle=" ", marker="o", color="red")

# Use the plt.show() method to display the figure
plt.show()

# ----------------------------------------------------------------------#

# Plotting the model on the data
# Pass times and measured distances into model

model_distances = model(times, measured_distances)

# Create figure and axis objects and call axis.plot() twice to plot data and model distances versus times
fig, axis = plt.subplots()
axis.plot(times, measured_distances, linestyle=" ", marker="o", color="black", label="Measured")
axis.plot(times, model_distances, linestyle="-", marker=None, color="red", label="Modeled")

# Add grid lines and a legend to your plot, and then show to display
axis.grid(True)
axis.legend(loc="best")
plt.show()

#--------------------------------------------------------------------------------------#

# Visually Estimating The Slope and Intercept
# Look at the plot data and guess initial trial values

trial_slope = 1
trial_intercept = 2

# input thoses guesses into the model function to compute the model values.
xm, ym = model(trial_intercept, trial_slope)

# Compare your your model to the data with the plot function
fig = plot_data_and_model(xd, yd, xm, ym)
plt.show()

# Repeat the steps above until your slope and intercept guess makes the model line up with the data.
final_slope = 1
final_intercept = 2

#--------------------------------------------------------------------------------------------#

# Quantifying Linear Relationships

#Mean, Deviation & Standard deviation
# Compute the deviations by subtracting the mean offset
dx = x - np.mean(x)
dy = y - np.mean(y)

# Normalize the data by dividing the deviations by the standard deviation
zx = dx / np.std(x)
zy = dy / np.std(y)

# Plot comparisons of the raw data and the normalized data
fig = plot_cdfs(dx, dy, zx, zy)

#-------------------------------------------------------------------------------------------#
# Covariance vs correlation
# Compute the covariance from the deviations.
dx = x - np.mean(x)
dy = y - np.mean(y)
covariance = np.mean(dx * dy)
print("Covariance: ", covariance)

# Compute the correlation from the normalized deviations.
zx = dx / np.std(x)
zy = dy / np.std(y)
correlation = np.mean(zx * zy)
print("Correlation: ", correlation)

# Plot the normalized deviations for visual inspection.
fig = plot_normalized_deviations(zx, zy)

#-----------------------------------------------------------------------------------#

#Correlation Stength
# Complete the function that will compute correlation.
def correlation(x,y):
    x_dev = x - np.mean(x)
    y_dev = y - np.mean(y)
    x_norm = x_dev / np.std(x)
    y_norm = y_dev / np.std(y)
    return np.mean(x_norm * y_norm)

# Compute and store the correlation for each data set in the list.
for name, data in data_sets.items():
    data['correlation'] = correlation(data['x'], data['y'])
    print('data set {} has correlation {:.2f}'.format(name, data['correlation']))

# Assign the data set with the best correlation.
best_data = data_sets['A']










