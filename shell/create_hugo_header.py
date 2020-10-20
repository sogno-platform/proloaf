import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--date', type=str, default=None)
parser.add_argument('--link', type=str, default=None)
parser.add_argument('--title', type=str, default=None)
parser.add_argument('--imginclude', default=False)

args = parser.parse_args(sys.argv[1:])

front_matter = "---\n"

if args.date:
    front_matter = front_matter + "date: " + args.date + "\n"

if args.title:
    front_matter = front_matter + "title: " + args.title + "\n"

if args.link:
    front_matter = front_matter + "linkTitle: " + args.link + "\n"

if args.imginclude:
    front_matter = front_matter + "resources:\n- src: \"./**.{png,jpg}\"\n  title: \"Image #:counter\"\n  params:\n    byline:\"Results of train run\"\n"

front_matter = front_matter + "---"

# front_matter = "---\ndate: "\
#                 + str(sys.argv[1])\
#                 + "\ntitle: \"" + str(sys.argv[2]) + " " + str(sys.argv[1])\
#                 + "\"\nlinkTitle: \"" + str(sys.argv[1])\
#                 + "\"\nresources:\n- src: \"./**.{png,jpg}\"\n  title: \"Image #:counter\"\n  params:\n    byline:\"Results of train run\"\n---"
print(front_matter)
