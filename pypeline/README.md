# German Shallow Discourse Parsing Pipeline

An under-development version of a tool for end-to-end shallow discourse parsing for German. 

Currently, only components for connective classification and argument extraction are implemented. 
Components for sense classification and implicit relations, as well as the tool itself, are still under development.

A screencast of an experimental GUI is available for download [here](annis.ling.uni-potsdam.de/gsdp_gui_demo.mp4). The user-specified input text is processed and connective (yellow) and argument (green and blue) tokens are highlighted according to parser output. Final visualisation issues (the amount of overlapping annotations for longer texts render individual annotations practically unreadable) as well as performance issues (processing time for longer texts) are all subject to ongoing developments.

## License

This code is released under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
You can find a human-readable summary of the licence agreement here:

https://creativecommons.org/licenses/by-nc-sa/4.0/

## References

The connective classification component is based on:</br>
Peter Bourgonje and Manfred Stede. "Identifying explicit discourse connectives in German." 
*In Proceedings of the 19th Annual SIGdial Meeting on Discourse and Dialogue, 327–331.* Melbourne, Australia, 2018. 
Association for Computational Linguistics. [PDF](http://aclweb.org/anthology/W18-5037) 

The argument extraction component is based on:</br>
Peter Bourgonje and Manfred Stede. "Explicit Discourse Argument Extraction for German." 
*In Proceedings of the 21st International Conference on Text, Speech and Dialogue.* Ljubljana, Slovenia, 2019. [PDF](https://link.springer.com/chapter/10.1007/978-3-030-27947-9_3)
