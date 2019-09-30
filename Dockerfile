# Copyright 2015 Guillaume Pellerin
# Copyright 2015 Thomas Fillon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM parisson/timeside:latest-dev

MAINTAINER Guillaume Pellerin <yomguy@parisson.com>, Thomas Fillon <thomas@parisson.com>

RUN if [ ! -d /srv/src/ ]; then mkdir /srv/src/; fi
RUN mkdir /srv/src/timeside-dummy
WORKDIR /srv/src/timeside-dummy

# Clone app
ADD . /srv/src/timeside-dummy/

RUN pip install pyyaml
RUN pip install librosa==0.7.0

# Install TimeSide Dummy
RUN pip install -e .
