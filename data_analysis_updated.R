pdf(NULL)

library(ggplot2)
library(dplyr)
library(gridExtra)

# The 'df' variable is injected directly by Python memory.
df$timestamp <- as.POSIXct(df$timestamp)

# Slice the timestamps into 5-second buckets
df$time_bin <- as.POSIXct(cut(df$timestamp, breaks = "5 secs"))

# 1. LINE PLOT (trend)
df_avg <- df %>%
  group_by(time_bin) %>%
  summarise(avg_people = mean(people, na.rm=TRUE)) %>%
  arrange(time_bin)

p1 <- ggplot(df_avg, aes(x = time_bin, y = avg_people, group = 1)) +
  geom_line(color = "blue", size=1) +
  theme_minimal() +
  labs(title = "Average Occupancy (5s)", x = "Time", y = "Avg People") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 2. DISTRIBUTION
p2 <- ggplot(df, aes(x = people)) +
  geom_histogram(bins = 10, fill = "steelblue", color="white") +
  theme_minimal() +
  labs(title = "Occupancy Distribution", x = "Number of People", y = "Frequency")

# 3. HEATMAP
p3 <- ggplot(df_avg, aes(x = time_bin, y = 1, fill = avg_people)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "red") +
  theme_minimal() +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Occupancy Heatmap (5s)", x = "Time", fill = "Avg")

# Save the combined layout to disk so OpenCV can pick it up
combined_plots <- arrangeGrob(p1, p2, p3, ncol=3)
ggsave("temp_r_plots.png", combined_plots, width=15, height=4, dpi=100)