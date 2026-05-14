library(ggplot2)
library(dplyr)

# read latest data
df <- read.csv("mock_data.csv")

# convert timestamp properly
df$timestamp <- as.POSIXct(df$timestamp)

# basic stats
avg_people <- mean(df$people)
peak_people <- max(df$people)

print(avg_people)
print(peak_people)

# extract time (minute-level grouping)
df$minute <- format(df$timestamp, "%H:%M")

df_avg <- df %>%
  group_by(minute) %>%
  summarise(avg_people = mean(people)) %>%
  arrange(minute)

# LINE PLOT (trend)
ggplot(df_avg, aes(x = minute, y = avg_people, group = 1)) +
  geom_line(color = "blue") +
  theme_minimal() +
  labs(title = "Average Occupancy Over Time",
       x = "Time (minute)",
       y = "Avg People")


# DISTRIBUTION (optional but good for report)
ggplot(df, aes(x = people)) +
  geom_histogram(bins = 10, fill = "steelblue") +
  theme_minimal() +
  labs(title = "Occupancy Distribution",
       x = "Number of People",
       y = "Frequency")


# HEATMAP (fixed version)
df_heat <- df %>%
  group_by(minute) %>%
  summarise(avg_people = mean(people))

ggplot(df_heat, aes(x = minute, y = 1, fill = avg_people)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "red") +
  theme_minimal() +
  theme(axis.text.y = element_blank(),
        axis.ticks.y = element_blank()) +
  labs(title = "Occupancy Heatmap",
       x = "Time",
       fill = "Avg People")