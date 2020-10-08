import sys

front_matter = "---\ndate: "\
                + str(sys.argv[1])\
                + "\ntitle: \"" + str(sys.argv[2])\
                + "\"\nlinkTitle: \"" + str(sys.argv[2]) + " " + str(sys.argv[1])\
                + "\"\nresources:\n- src: \"./**.{png,jpg}\"\n  title: \"Image #:counter\"\n  params:\n    byline:\"Results of train run\"\n---"
print(front_matter)
