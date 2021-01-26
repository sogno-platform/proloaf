# Copyright 2021 The ProLoaF Authors. All Rights Reserved.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# ==============================================================================

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
