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


def create_frontmatter(date = None, link = None, title = None, imginclude=False):
    front_matter = "---\n"

    if date:
        front_matter = front_matter + "date: " + date + "\n"

    if title:
        front_matter = front_matter + "title: " + title + "\n"

    if link:
        front_matter = front_matter + "linkTitle: " + link + "\n"

    if imginclude:
        front_matter = front_matter + "resources:\n- src: \"./**.{png,jpg}\"\n  title: \"Image #:counter\"\n  params:\n    byline:\"Results of train run\"\n"

    front_matter = front_matter + "---"

    return front_matter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default=None)
    parser.add_argument('--link', type=str, default=None)
    parser.add_argument('--title', type=str, default=None)
    parser.add_argument('--imginclude', default=False)

    args = parser.parse_args(sys.argv[1:])

    print(create_frontmatter(args.date,args.title,args.link,args.imginclude))
