import pstats 
p = pstats.Stats("speed.txt")
p.sort_stats("cumulative").print_stats(100)