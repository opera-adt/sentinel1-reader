# release history
import collections

# release history
Tag = collections.namedtuple("Tag", "version date")
release_history = (
    Tag("0.2.4", "2024-03-11"),
    Tag("0.2.3", "2023-09-21"),
    Tag("0.2.2", "2023-09-08"),
    Tag("0.2.1", "2023-08-23"),
    Tag("0.2.0", "2023-07-25"),
    Tag("0.1.7", "2023-05-09"),
    Tag("0.1.6", "2023-03-22"),
    Tag("0.1.5", "2022-12-21"),
    Tag("0.1.4", "2022-12-16"),
    Tag("0.1.3", "2022-12-15"),
    Tag("0.1.2", "2022-07-22"),
    Tag("0.1.1", "2022-07-21"),
    Tag("0.1.0", "2022-06-07"),
)

# latest release version number and date
release_version = release_history[0].version
release_date = release_history[0].date
