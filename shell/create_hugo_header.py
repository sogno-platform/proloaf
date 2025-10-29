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

# Currently only being used for the results page generation
"""
Create Hugo frontmatter and content with dynamic image sections
This script generates Hugo frontmatter and content sections based on images found in a specified directory.
It categorizes images based on their filenames and generates a structured markdown output.
"""

import argparse
import glob
import os


def find_images_in_directory(base_path):
    """Find all images in the directory structure and return categorized images information."""
    if not os.path.exists(base_path):
        return {}

    image_info = {}

    # Find all PNG and JPG files recursively
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        pattern = os.path.join(base_path, "**", ext)
        images = glob.glob(pattern, recursive=True)

        for img_path in images:
            # Get relative path from base_path
            rel_path = os.path.relpath(img_path, base_path)
            filename = os.path.basename(img_path)
            name_without_ext = os.path.splitext(filename)[0]

            # Organize by type/category based on filename patterns
            category = categorize_image(name_without_ext)

            if category not in image_info:
                image_info[category] = []

            image_info[category].append(
                {
                    "path": rel_path,
                    "filename": filename,
                    "name": name_without_ext,
                    "display_name": generate_display_name(name_without_ext, rel_path),
                }
            )

    return image_info


def categorize_image(filename):
    """Categorize images based on filename patterns"""
    filename_lower = filename.lower()

    if "hour" in filename_lower or "eval_hour" in filename_lower:
        return "hourly"
    elif "metric" in filename_lower:
        return "metrics"
    elif "boxplot" in filename_lower or "box" in filename_lower:
        return "plots"
    elif "summary" in filename_lower or "overview" in filename_lower:
        return "summary"
    else:
        return "other"


def generate_display_name(filename, rel_path):
    """Generate a human-readable display name"""
    # Extract meaningful parts from filename and path
    parts = rel_path.split("/")

    # Remove common prefixes
    clean_name = filename.replace("eval_", "").replace("_", " ")

    # Add context from directory structure if helpful
    if len(parts) > 1:
        parent_dir = parts[-2]  # Get parent directory
        if parent_dir not in clean_name.lower():
            clean_name = f"{parent_dir} - {clean_name}"

    return clean_name.title()


def create_frontmatter(date=None, link=None, title=None, imginclude=False, model_path=None):
    front_matter = "---\n"

    if date:
        front_matter += f"date: {date}\n"

    if title:
        front_matter += f"title: {title}\n"

    if link:
        front_matter += f"linkTitle: {link}\n"

    if imginclude:
        front_matter += 'resources:\n- src: "**/*.{png,jpg,jpeg}"\n  title: "Image #:counter"\n  params:\n    byline: "Results of train run"\n'

    front_matter += "---\n"

    return front_matter


def generate_content_with_images(model_path, pipeline_id=None, station=None):
    """Generate the complete markdown content with dynamic image sections"""
    content = ""

    if pipeline_id:
        content += f"[Source Pipeline](https://git.rwth-aachen.de/acs/public/automation/plf/proloaf/-/pipelines/{pipeline_id})\n\n"

    content += "**CI Pipeline Result page**\n\n"

    # Find and categorize images
    image_info = find_images_in_directory(model_path)

    if not image_info:
        content += "*No images found in the model directory.*\n"
        return content

    # Generate sections for different categories
    category_order = ["hourly", "metrics", "plots", "summary", "other"]
    category_titles = {
        "hourly": "Hourly Forecasts",
        "metrics": "Performance Metrics",
        "plots": "Analysis Plots",
        "summary": "Summary Results",
        "other": "Additional Results",
    }

    for category in category_order:
        if category in image_info:
            if len(image_info) > 1:  # Only add category headers if multiple categories
                content += f"## {category_titles[category]}\n\n"

            # Sort images within category
            sorted_images = sorted(image_info[category], key=lambda x: x["name"])

            for img in sorted_images:
                content += f'{{{{< imgproc "{img["path"]}" Fill "500x300" >}}}}\n'
                content += f'{img["display_name"]}\n'
                content += "{{< /imgproc >}}\n\n"

    return content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Hugo frontmatter and content")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--link", type=str, default=None)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--imginclude", action="store_true", default=False)
    parser.add_argument(
        "--model-path", type=str, default=None, help="Path to model directory for dynamic content generation"
    )
    parser.add_argument("--pipeline-id", type=str, default=None)
    parser.add_argument("--station", type=str, default=None)
    parser.add_argument(
        "--content-only", action="store_true", default=False, help="Generate only content, not frontmatter"
    )

    args = parser.parse_args()

    if args.content_only and args.model_path:
        # Generate only content with images
        print(generate_content_with_images(args.model_path, args.pipeline_id, args.station))
    elif args.model_path:
        # Generate frontmatter + content
        frontmatter = create_frontmatter(args.date, args.link, args.title, args.imginclude)
        content = generate_content_with_images(args.model_path, args.pipeline_id, args.station)
        print(frontmatter)
        print(content)
    else:
        # Generate only frontmatter
        print(create_frontmatter(args.date, args.link, args.title, args.imginclude))
